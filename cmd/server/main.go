package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"time"

	pb "github.com/arhantsg07/ml-inference-system/proto/inference"
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
		return nil, fmt.Errorf("error marshaling json: %v", err)
	}

	// logging
	log.Printf("Sending request to %s", apiURL)
	log.Printf("Request body: %s", string(jsonData))

	// sending the http post req
	req, err := http.NewRequestWithContext(ctx, "POST", apiURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, status.Errorf(
			codes.InvalidArgument,
			"Failed to create external API request",
		)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	resp, err := s.httpClient.Do(req)
	if err != nil {
		return nil, status.Errorf(
			codes.Unavailable,
			"Failed to reach external API", // error seding request
		)
	}
	defer resp.Body.Close()

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, status.Errorf(
			codes.Unavailable,
			"failed to read response from external API",
		)
	}

	log.Printf("API Response Status: %d", resp.StatusCode)
	log.Printf("API Response Body: %s", string(body))

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("API returned status %d: %s", resp.StatusCode, string(body))
	}

	var apiResponse APIResponse
	if err := json.Unmarshal(body, &apiResponse); err != nil {
		return nil, status.Errorf(
			codes.Internal,
			"Failed to parse external API response",
		)
	}

	return &apiResponse, nil

}

// Predict takes the input data and then calls the sendDataToAPI function.
func (s *server) Predict(ctx context.Context, req *pb.PredictRequest) (*pb.PredictResponse, error) {
	var inputArray []float64

	if err := json.Unmarshal(req.GetInputData(), &inputArray); err != nil {
		log.Printf("failed to unmarshal input: %v", err)

		return nil, status.Errorf(
			codes.InvalidArgument,
			"input_data must be a JSON array of numbers",
		)
	}
	// inputArray := req.GetInputData()
	// if len(inputArray) == 0 {
	// 	return nil, status.Errorf(
	// 		codes.InvalidArgument, "input data cannot be empty",
	// 	)
	// }

	log.Printf("Parsed input array: %v", inputArray)

	// referring to the above struct
	input_data := &InputData{
		ModelName: req.GetModelName(),
		Input:     inputArray,
	}

	apiResponse, err := s.sendDataToAPI(ctx, input_data)
	if err != nil {
		log.Printf("Error sending to external API: %v", err)
		return nil, status.Errorf(
			codes.Unavailable,
			"failed to call external API",
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
		return nil, status.Errorf(
			codes.Internal,
			"output empty",
		)
	}

	// if len(outputBytes) == 0 {
	// 	return nil, status.Errorf(
	// 		codes.Internal,
	// 		"output empty",
	// 	)
	// }

	// else return the formatted response to the client

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

	s := grpc.NewServer()
	pb.RegisterInferenceServer(s, &server{
		httpClient: httpClient,
	})
	log.Printf("gRPC Inference server listening on %s", *port)
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
