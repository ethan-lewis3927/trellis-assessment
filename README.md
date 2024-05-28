# trellis-assessment

## Description
Model and API code for tellis take home assesment

## Setup

### 1. Clone the Repository
Clone this repository to your local machine using the following command:
```bash
git clone https://github.com/ethan-lewis3927/trellis-assessment
```

### 2. Train the model
From your the root directory run
```python
python3 -m trainer.trainer
```

### 2. Start the API
From your the root directory run
```python
python3 -m API.app
```

## API Usage

### Endpoint: /classify_document
Method: POST

Request Body:
```json
{
  "document_text": "string, required"
}
```

### Success Response:
Code: 200 OK
```json
{
  "message": "success",
  "predicted_label": "business",
  "confidence": 0.85
}
```

### Error Responses:
Code: 400 Bad Request
```json
{
  "error": "Missing 'document_text' in request body"
}
```

Code: 500 Internal Server Error
```json
{
  "error": "Description of the error"
}
```
