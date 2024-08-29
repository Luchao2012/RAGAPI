from fastapi.testclient import TestClient
from main import app, QueryInput

client = TestClient(app)

def test_qa_endpoint():
    # Prepare the request data
    request_data = {
        "query": "what is well logging?",
        "llm": "llm_gpt4",
        "filter_top_n": 5,
        "similarity_top_k": 30,
        "semantic_weight": 0.7,
        "collection_name": "rag_exp",
        "persist_dir": "../user_db",
        "use_rerank": True,
        "rerank_top_n": 15
    }

    # Send the POST request
    response = client.post("/qa/", json=request_data)
    
    # Print the response status code and JSON body
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())

    # Optionally, add assertions to verify the response is as expected
    assert response.status_code == 200
    assert "response_text" in response.json()
    assert "source_metadata" in response.json()
    
    # You can add more assertions based on your expected response

if __name__ == "__main__":
    test_qa_endpoint()
