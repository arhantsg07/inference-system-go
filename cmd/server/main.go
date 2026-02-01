package main

import (
	"context"
	"flag"
	"log"
	"net"
	pb "github.com/arhantsg07/ml-inference-system/proto/inference"
	"google.golang.org/grpc"
)

var (
	port = flag.String("port", ":50051", "Server port, include ':' e.g. :50051")
)

// server implements the Inference gRPC service.
type server struct {
	pb.UnimplementedInferenceServer
}

// Predict is a dummy implementation that echoes back a static response.
func (s *server) Predict(ctx context.Context, req *pb.PredictRequest) (*pb.PredictResponse, error) {
	log.Printf("Received Predict request for model=%s input_size=%d", req.ModelName, len(req.InputData))
	// TODO: replace with real model inference using ONNX Runtime / model runner.
	return &pb.PredictResponse{
		OutputData: []byte("dummy-output"),
		Status:     "ok",
	}, nil
}

func main() {
	flag.Parse()

	lis, err := net.Listen("tcp", *port)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	s := grpc.NewServer()
	pb.RegisterInferenceServer(s, &server{})
	log.Printf("gRPC Inference server listening on %s", *port)
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
