package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	pb "github.com/arhantsg07/ml-inference-system/proto/inference"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

var (
	port = flag.String("port", ":50051", "Server port, include ':' e.g. :50051")
)

// server implements the Inference gRPC service.
type server struct {
	pb.UnimplementedInferenceServer
	httpClient *http.Client
}

var (
	requestCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "inference_requests_total",
			Help: "Total number of inference requests",
		},
		[]string{"method", "status"},
	)
	requestDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "inference_request_duration_seconds",
			Help:    "Histogram of inference request latencies (seconds)",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"method"},
	)
)

func init() {
	prometheus.MustRegister(requestCount, requestDuration)
}

type InputData struct {
	ModelName string    `json:"model_name"`
	Input     []float64 `json:"input"`
}

type APIResponse struct {
	ModelName string    `json:"model_name"`
	Output    []float64 `json:"output"`
	Status    string    `json:"status"`
}

func (s *server) sendDataToAPI(ctx context.Context, inputData *InputData) (*APIResponse, error) {
	apiURL := "http://localhost:8080/predict"

	requestBody := InputData{
		ModelName: inputData.ModelName,
		Input:     inputData.Input,
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, status.Errorf(
			codes.Internal,
			"error marshaling json: %v", err,
		)
	}

	// logging (trim long bodies in production)
	log.Printf("Sending request to %s", apiURL)
	if len(jsonData) < 4096 {
		log.Printf("Request body: %s", string(jsonData))
	} else {
		log.Printf("Request body too large to print (%d bytes)", len(jsonData))
	}

	// sending the http post req with context from gRPC
	req, err := http.NewRequestWithContext(ctx, "POST", apiURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, status.Errorf(
			codes.InvalidArgument,
			"Failed to create external API request: %v", err,
		)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	resp, err := s.httpClient.Do(req)
	if err != nil {
		return nil, status.Errorf(
			codes.Unavailable,
			"Failed to reach external API: %v", err,
		)
	}
	defer resp.Body.Close()

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, status.Errorf(
			codes.Unavailable,
			"failed to read response from external API: %v", err,
		)
	}

	log.Printf("API Response Status: %d", resp.StatusCode)
	if len(body) < 4096 {
		log.Printf("API Response Body: %s", string(body))
	} else {
		log.Printf("API response body too large to print (%d bytes)", len(body))
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		// Map 4xx to InvalidArgument, 5xx to Internal/Unavailable
		if resp.StatusCode >= 400 && resp.StatusCode < 500 {
			return nil, status.Errorf(codes.InvalidArgument, "API returned status %d: %s", resp.StatusCode, string(body))
		}
		return nil, status.Errorf(codes.Internal, "API returned status %d: %s", resp.StatusCode, string(body))
	}

	var apiResponse APIResponse
	if err := json.Unmarshal(body, &apiResponse); err != nil {
		return nil, status.Errorf(
			codes.Internal,
			"Failed to parse external API response: %v", err,
		)
	}

	return &apiResponse, nil

}

// Predict takes the input data and then calls the sendDataToAPI function.
func (s *server) Predict(ctx context.Context, req *pb.PredictRequest) (*pb.PredictResponse, error) {
	start := time.Now()
	method := "Predict"
	var statusLabel string = "ok"
	defer func() {
		requestDuration.WithLabelValues(method).Observe(time.Since(start).Seconds())
		requestCount.WithLabelValues(method, statusLabel).Inc()
	}()

	var inputArray []float64

	if err := json.Unmarshal(req.GetInputData(), &inputArray); err != nil {
		log.Printf("failed to unmarshal input: %v", err)
		statusLabel = "bad-input"

		return nil, status.Errorf(
			codes.InvalidArgument,
			"input_data must be a JSON array of numbers",
		)
	}

	if len(inputArray) == 0 {
		statusLabel = "empty-input"
		return nil, status.Errorf(
			codes.InvalidArgument, "input data cannot be empty",
		)
	}

	log.Printf("Parsed input array: %v", inputArray)

	// referring to the above struct
	input_data := &InputData{
		ModelName: req.GetModelName(),
		Input:     inputArray,
	}

	apiResponse, err := s.sendDataToAPI(ctx, input_data)
	if err != nil {
		log.Printf("Error sending to external API: %v", err)
		statusLabel = "api-error"
		return nil, status.Errorf(
			codes.Unavailable,
			"failed to call external API: %v", err,
		)
	}

	log.Printf("Successfully sent data to external API")
	log.Printf("Successfully processed the prediction request")

	log.Printf("Model: %s, Output: %v, Status: %s",
		apiResponse.ModelName, apiResponse.Output, apiResponse.Status)

	// converting the response to match the gRPC format
	// throw err, if failed marshalling
	outputBytes, err := json.Marshal(apiResponse.Output)
	if err != nil {
		statusLabel = "internal-error"
		return nil, status.Errorf(
			codes.Internal,
			"failed to marshal output: %v", err,
		)
	}
	return &pb.PredictResponse{
		OutputData: outputBytes,
		Status:     apiResponse.Status,
	}, nil
}

func main() {
	flag.Parse()

	lis, err := net.Listen("tcp", *port)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	httpClient := &http.Client{
		Timeout: 10 * time.Second,
	}

	grpcServer := grpc.NewServer()
	pb.RegisterInferenceServer(grpcServer, &server{
		httpClient: httpClient,
	})

	// Start HTTP server for /metrics and /health
	httpMux := http.NewServeMux()
	httpMux.Handle("/metrics", promhttp.Handler())
	httpMux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ok"))
	})

	httpSrv := &http.Server{
		Addr:    ":9090",
		Handler: httpMux,
	}

	// Run HTTP server in background
	go func() {
		log.Printf("HTTP metrics server listening on %s", httpSrv.Addr)
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("HTTP server ListenAndServe: %v", err)
		}
	}()

	// Run gRPC server in background
	go func() {
		log.Printf("gRPC Inference server listening on %s", *port)
		if err := grpcServer.Serve(lis); err != nil {
			log.Fatalf("failed to serve gRPC: %v", err)
		}
	}()

	// Handle graceful shutdown
	stop := make(chan os.Signal, 1)						// makes a memory allocation for receiving signal
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)  // registers the interest in the signals interrupt, sigterm
	<-stop												// waits for the signal
	log.Printf("Shutting down servers...")

	// Shutdown HTTP server with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := httpSrv.Shutdown(ctx); err != nil {
		log.Printf("HTTP server Shutdown: %v", err)
	}

	// Gracefully stop gRPC server; give it some time then force stop
	stopped := make(chan struct{})
	go func() {
		grpcServer.GracefulStop()
		close(stopped)
	}()

	select {
	case <-stopped:
		log.Printf("gRPC server stopped gracefully")
	case <-time.After(10 * time.Second):
		log.Printf("gRPC server did not stop in time; forcing stop")
		grpcServer.Stop()
	}

	log.Printf("Shutdown complete")
}
