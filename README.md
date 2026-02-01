# ML Inference System

A minimal, production‑oriented **gRPC‑based ML inference service** written in Go.
This project is structured using standard Go conventions and is designed to be easy to extend with real ML models, multiple services, and deployment tooling.

---

## 📁 Project Structure

```
ml-inference-system
├── README.md
├── cmd
│   ├── client
│   │   └── main.go
│   └── server
│       └── main.go
├── go.mod
├── go.sum
└── proto
    └── inference
        ├── inference.pb.go
        ├── inference.proto
        └── inference_grpc.pb.go

```

---

## 🚀 Getting Started

### Prerequisites

* Go **1.21+** (recommended: latest stable)
* `protoc` (Protocol Buffers compiler)
* Go protobuf plugins

Install `protoc` plugins (one‑time):

```bash
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
```

Make sure `$GOPATH/bin` is in your `PATH`.

---

## 📦 Initialize the Project

Initialize the Go module:

```bash
go mod init github.com/<your-username>/ml-inference-system
```

Install dependencies:

```bash
go get google.golang.org/grpc
go get google.golang.org/protobuf
go mod tidy
```

---

## 🧬 Protobuf Definition

The protobuf file lives in:

```
proto/inference/inference.proto
```

It must include a `go_package` option matching the module path:

```proto
syntax = "proto3";

package inference;

option go_package = "github.com/<your-username>/ml-inference-system/proto/inference;inference";
```

---

## 🔧 Generate gRPC Code

Run from the **project root**:

```bash
protoc --go_out=. --go_opt=paths=source_relative \
    --go-grpc_out=. --go-grpc_opt=paths=source_relative \
    proto/inference/inference.proto
```

Generated files will appear in:

```
proto/inference/
```

---

## 🖥 Running the Server

From the project root:

```bash
cd cmd/server
go run main.go
```

The server will start listening on the configured gRPC port (see `main.go`).

---

## 🖥 Running the client

From the project root:

```bash
cd cmd/client
go run main.go
```

The client will send the request on the configured gRPC port (see `main.go`).

---

## 📥 Importing the Protobuf Package

In Go code:

```go
import pb "github.com/<your-username>/ml-inference-system/proto/inference"
```

---
