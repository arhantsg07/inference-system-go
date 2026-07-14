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
	"strings"
	"syscall"
	"time"
	"fmt"
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
	ModelName string `json:"model_name"`
	Input     string `json:"input"`
}

type APIResponse struct {
	ModelName        string         `json:"model_name"`
	Output           []interface{}  `json:"output"`
	Status           string         `json:"status"`
	InferenceTimeMs  float64        `json:"inference_time_ms"`
}

func (s *server) sendDataToAPI(ctx context.Context, inputData *InputData) (*APIResponse, error) {
	baseURL := os.Getenv("MODEL_SERVER_URL")
	if baseURL == "" {
		baseURL = "http://localhost:8080"
	}
	apiURL := fmt.Sprintf("%s/predict", strings.TrimRight(baseURL, "/"))

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

	log.Printf("Sending request to %s (model=%s, body=%d bytes)", apiURL, inputData.ModelName, len(jsonData))

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

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, status.Errorf(
			codes.Unavailable,
			"failed to read response from external API: %v", err,
		)
	}

	log.Printf("API Response Status: %d, body=%d bytes", resp.StatusCode, len(body))

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
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

func (s *server) Predict(ctx context.Context, req *pb.PredictRequest) (*pb.PredictResponse, error) {
	start := time.Now()
	method := "Predict"
	var statusLabel string = "ok"
	defer func() {
		requestDuration.WithLabelValues(method).Observe(time.Since(start).Seconds())
		requestCount.WithLabelValues(method, statusLabel).Inc()
	}()

	var requestData InputData

	if err := json.Unmarshal(req.GetInputData(), &requestData); err != nil {
		log.Printf("failed to unmarshal input: %v", err)
		statusLabel = "bad-input"

		return nil, status.Errorf(
			codes.InvalidArgument,
			"input_data must be a JSON object with model_name and input fields",
		)
	}

	if requestData.ModelName == "" {
		statusLabel = "empty-model"
		return nil, status.Errorf(
			codes.InvalidArgument, "model_name cannot be empty",
		)
	}

	if requestData.Input == "" {
		statusLabel = "empty-input"
		return nil, status.Errorf(
			codes.InvalidArgument, "input cannot be empty",
		)
	}

	log.Printf("Predict request: model=%s, input_size=%d bytes", requestData.ModelName, len(requestData.Input))

	apiResponse, err := s.sendDataToAPI(ctx, &requestData)
	if err != nil {
		log.Printf("Error sending to external API: %v", err)
		statusLabel = "api-error"
		return nil, status.Errorf(
			codes.Unavailable,
			"failed to call external API: %v", err,
		)
	}

	log.Printf("Prediction: model=%s, detections=%d, inference_time=%.2fms, status=%s",
		apiResponse.ModelName, len(apiResponse.Output), apiResponse.InferenceTimeMs, apiResponse.Status)

	// include inference_time_ms in output data
	enhancedOutput := map[string]interface{}{
		"detections":       apiResponse.Output,
		"inference_time_ms": apiResponse.InferenceTimeMs,
	}
	enhancedBytes, err := json.Marshal(enhancedOutput)
	if err != nil {
		statusLabel = "internal-error"
		return nil, status.Errorf(
			codes.Internal,
			"failed to marshal enhanced output: %v", err,
		)
	}

	return &pb.PredictResponse{
		OutputData: enhancedBytes,
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
		Timeout: 30 * time.Second,
	}

	grpcServer := grpc.NewServer()
	pb.RegisterInferenceServer(grpcServer, &server{
		httpClient: httpClient,
	})

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

	go func() {
		log.Printf("HTTP metrics server listening on %s", httpSrv.Addr)
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("HTTP server ListenAndServe: %v", err)
		}
	}()

	go func() {
		log.Printf("gRPC Inference server listening on %s", *port)
		if err := grpcServer.Serve(lis); err != nil {
			log.Fatalf("failed to serve gRPC: %v", err)
		}
	}()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	<-stop
	log.Printf("Shutting down servers...")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := httpSrv.Shutdown(ctx); err != nil {
		log.Printf("HTTP server Shutdown: %v", err)
	}

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
