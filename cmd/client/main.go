package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"flag"
	"log"
	"os"
	"time"

	pb "github.com/arhantsg07/ml-inference-system/proto/inference"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

var (
	serverAddr = flag.String("addr", "localhost:50051", "The server address in the format of host:port")
)

type InputData struct {
	ModelName string `json:"model_name"`
	Input     string `json:"input"`
}

func MakePrediction(client pb.InferenceClient, req *pb.PredictRequest) {
	log.Printf("Sending prediction request for model %s...", req.ModelName)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	prediction, err := client.Predict(ctx, req)
	if err != nil {
		log.Printf("client.Predict failed: %v", err)
		return
	}
	log.Printf("Response status: %s", prediction.Status)
	log.Printf("Response output: %s", string(prediction.OutputData))
}

func main() {
	flag.Parse()

	var opts []grpc.DialOption
	opts = append(opts, grpc.WithTransportCredentials(insecure.NewCredentials()))
	conn, err := grpc.NewClient(*serverAddr, opts...)
	if err != nil {
		log.Fatalf("fail to dial: %v", err)
	}
	defer conn.Close()
	client := pb.NewInferenceClient(conn)

	imagePath := "models/sample_test.jpg"
	if len(flag.Args()) > 0 {
		imagePath = flag.Args()[0]
	}

	imageBytes, err := os.ReadFile(imagePath)
	if err != nil {
		log.Fatalf("Failed to read image file: %v", err)
	}

	base64Image := base64.StdEncoding.EncodeToString(imageBytes)
	log.Printf("Read image %s (%d bytes, base64=%d chars)", imagePath, len(imageBytes), len(base64Image))

	inputData := InputData{
		ModelName: "yolov8n",
		Input:     base64Image,
	}

	inputBytes, err := json.Marshal(inputData)
	if err != nil {
		log.Fatalf("Error marshaling input: %v", err)
	}

	MakePrediction(client, &pb.PredictRequest{
		ModelName: "yolov8n",
		InputData: inputBytes,
	})
}
