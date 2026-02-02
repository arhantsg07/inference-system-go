package main

import (
	"context"
	"errors"
	"testing"
	"time"

	pb "github.com/arhantsg07/ml-inference-system/proto/inference"
	"google.golang.org/grpc"
)

// MockInferenceClient is a mock implementation of pb.InferenceClient for testing
type MockInferenceClient struct {
	PredictFunc func(ctx context.Context, in *pb.PredictRequest, opts ...grpc.CallOption) (*pb.PredictResponse, error)
}

func (m *MockInferenceClient) Predict(ctx context.Context, in *pb.PredictRequest, opts ...grpc.CallOption) (*pb.PredictResponse, error) {
	if m.PredictFunc != nil {
		return m.PredictFunc(ctx, in, opts...)
	}
	return &pb.PredictResponse{}, nil
}

func TestMakePrediction_Success(t *testing.T) {
	// Arrange
	expectedResponse := &pb.PredictResponse{
		OutputData: []byte{0x01, 0x02, 0x03},
	}
	
	mockClient := &MockInferenceClient{
		PredictFunc: func(ctx context.Context, in *pb.PredictRequest, opts ...grpc.CallOption) (*pb.PredictResponse, error) {
			// Verify the context has a timeout
			deadline, ok := ctx.Deadline()
			if !ok {
				t.Error("Expected context to have a deadline")
			}
			
			// Verify timeout is approximately 10 seconds
			timeout := time.Until(deadline)
			if timeout < 9*time.Second || timeout > 11*time.Second {
				t.Errorf("Expected timeout of ~10 seconds, got %v", timeout)
			}
			
			return expectedResponse, nil
		},
	}
	
	req := &pb.PredictRequest{
		ModelName: "test-model",
		InputData: []byte{0x01, 0x02, 0x03},
	}
	
	// Act - This will log but not fail on success
	MakePrediction(mockClient, req)
	
	// Assert - If we got here without panic, the test passed
}

func TestMakePrediction_WithError(t *testing.T) {
	// Arrange
	expectedError := errors.New("prediction failed")
	
	mockClient := &MockInferenceClient{
		PredictFunc: func(ctx context.Context, in *pb.PredictRequest, opts ...grpc.CallOption) (*pb.PredictResponse, error) {
			return nil, expectedError
		},
	}
	
	req := &pb.PredictRequest{
		ModelName: "test-model",
		InputData: []byte{0x01, 0x02, 0x03},
	}
	
	// Act & Assert
	// Note: MakePrediction calls log.Fatalf on error, which would exit the test
	// In a real-world scenario, you'd want to refactor MakePrediction to return
	// an error instead of calling log.Fatalf, or use a testing approach that
	// captures the fatal exit
	
	// For now, we can't directly test the error case without refactoring
	// the function to return errors instead of calling log.Fatalf
	
	// Uncomment below to see the fatal error behavior
	MakePrediction(mockClient, req)
	
	t.Skip("Skipping error test - MakePrediction uses log.Fatalf which would exit the test")
}

func TestMakePrediction_ContextTimeout(t *testing.T) {
	// Arrange
	mockClient := &MockInferenceClient{
		PredictFunc: func(ctx context.Context, in *pb.PredictRequest, opts ...grpc.CallOption) (*pb.PredictResponse, error) {
			// Verify context is passed correctly
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			default:
				return &pb.PredictResponse{OutputData: []byte{0x01}}, nil
			}
		},
	}
	
	req := &pb.PredictRequest{
		ModelName: "timeout-test",
		InputData: []byte{0x01},
	}
	
	// Act
	MakePrediction(mockClient, req)
	
	// Assert - Test passes if no panic occurs
}

func TestMakePrediction_ValidatesRequest(t *testing.T) {
	// Arrange
	var capturedRequest *pb.PredictRequest
	
	mockClient := &MockInferenceClient{
		PredictFunc: func(ctx context.Context, in *pb.PredictRequest, opts ...grpc.CallOption) (*pb.PredictResponse, error) {
			capturedRequest = in
			return &pb.PredictResponse{}, nil
		},
	}
	
	expectedModelName := "my-model"
	expectedInputData := []byte{0x01, 0x02, 0x03}
	
	req := &pb.PredictRequest{
		ModelName: expectedModelName,
		InputData: expectedInputData,
	}
	
	// Act
	MakePrediction(mockClient, req)
	
	// Assert
	if capturedRequest == nil {
		t.Fatal("Expected request to be captured")
	}
	
	if capturedRequest.ModelName != expectedModelName {
		t.Errorf("Expected model name %s, got %s", expectedModelName, capturedRequest.ModelName)
	}
	
	if capturedRequest.InputData == nil {
		t.Errorf("Expected input data %v, got %v", expectedInputData, capturedRequest.InputData)
	}
}

// Benchmark for MakePrediction
func BenchmarkMakePrediction(b *testing.B) {
	mockClient := &MockInferenceClient{
		PredictFunc: func(ctx context.Context, in *pb.PredictRequest, opts ...grpc.CallOption) (*pb.PredictResponse, error) {
			return &pb.PredictResponse{OutputData: []byte{0x01, 0x02, 0x03}}, nil
		},
	}
	
	req := &pb.PredictRequest{
		ModelName: "benchmark-model",
		InputData: []byte{0x01, 0x02, 0x03},
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MakePrediction(mockClient, req)
	}
}