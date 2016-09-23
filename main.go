package main

import (
	"os"
	"flag"
	"log"
	"context"
	"fmt"
	"io/ioutil"
	"encoding/base64"
	"encoding/json"

	vision "google.golang.org/api/vision/v1"
	"golang.org/x/oauth2/google"
)

const (
	SUCCESS = 0
	FAILURE = 1
)

func newVisionService() (*vision.Service, error) {
	fn := os.Getenv("GOOGLE_APPLICATION_CREDENTIALS")
	if len(fn) == 0 {
		return nil, fmt.Errorf("Unable to get env variable: GOOGLE_APPLICATION_CREDENTIALS")
	}

	b, err := ioutil.ReadFile(fn)
	if err != nil {
		return nil, fmt.Errorf("Unable to read client secret file: %v", err)
	}

	config, err := google.JWTConfigFromJSON(b, vision.CloudPlatformScope)
	if err != nil {
		return nil, fmt.Errorf("Unable to parse client secret file to config: %v", err)
	}

	ctx := context.Background()

	client := config.Client(ctx)

	srv, err := vision.New(client)
	if err != nil {
		return nil, fmt.Errorf("Unable to retrieve vision Client %v", err)
	}

	return srv, nil
}

func annotateImageRequest(path string, featureType string) (*vision.AnnotateImageRequest, error) {
	b, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("Unable to read an image by file path: %v", err)
	}

	req := &vision.AnnotateImageRequest{
		Image : &vision.Image{
			Content: base64.StdEncoding.EncodeToString(b),
		},
		Features: []*vision.Feature{
			{
				MaxResults: 10,
				Type: featureType,
			},
		},
	}

	return req, nil
}

func Run() int {
	var featureType string
	flag.StringVar(&featureType, "t", "FACE_DETECTION", "Type of image feature.")
	flag.Parse()
	args := flag.Args()
	if len(args) == 0 {
		log.Printf("Argument is required.")
		return FAILURE
	}

	srv, err := newVisionService()
	if err != nil {
		log.Printf("Unable to retrieve vision service: %v\n", err)
		return FAILURE
	}

	req, err := annotateImageRequest(args[len(args) - 1], featureType)
	if err != nil {
		log.Printf("Unable to retrieve image request: %v\n", err)
		return FAILURE
	}

	batch := &vision.BatchAnnotateImagesRequest{
		Requests:[]*vision.AnnotateImageRequest{req},
	}

	res, err := srv.Images.Annotate(batch).Do()
	if err != nil {
		log.Printf("Unable to execute images annotate requests: %v\n", err)
		return FAILURE
	}

	body, err := json.MarshalIndent(res.Responses, "", " ")
	if err != nil {
		log.Printf("Unable to marshal the response: %v\n", err)
		return FAILURE
	}
	fmt.Println(string(body))

	return SUCCESS;
}

func main() {
	os.Exit(Run())
}