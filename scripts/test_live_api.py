import time
import httpx
import sqlite3

def run_test():
    print("Waiting for server to be fully ready...")
    time.sleep(3)

    # 1. Check health
    health_url = "http://localhost:8000/health"
    try:
        r = httpx.get(health_url)
        print(f"Health check status: {r.status_code}, body: {r.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return

    # 2. Extract benchmark (POST /api/v1/benchmarks/analyze)
    job_description = (
        "We are hiring a Senior Backend Engineer with 3-5 years of Python experience. "
        "You will design distributed systems and RESTful APIs using FastAPI and Docker. "
        "Nice to have: GraphQL knowledge."
    )
    analyze_url = "http://localhost:8000/api/v1/benchmarks/analyze"
    print("\nSending job description to analyze endpoint...")
    try:
        r = httpx.post(
            analyze_url,
            content=job_description,
            headers={"Content-Type": "text/plain"},
            timeout=120.0
        )
        print(f"Analyze response status: {r.status_code}")
        if r.status_code != 201:
            print(f"Failure response: {r.text}")
            return
        
        data = r.json()
        benchmark_id = data["id"]
        print(f"Benchmark created successfully! ID: {benchmark_id}")
        print(f"Extracted Must Have Skills: {data['must_have_skills']}")
        print(f"Extracted Nice To Have Skills: {data['nice_to_have_skills']}")
        print(f"Extracted Tools: {data['required_tools']}")
        print(f"Extracted Experience Level: {data['experience_patterns']}")
    except Exception as e:
        print(f"Analyze request failed: {e}")
        return

    # 3. Query the local SQLite database to check the stored embedding vector length
    print("\nQuerying SQLite database to verify stored embedding vector length...")
    db_path = "career_coach_live_test.db"
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, embedding FROM role_benchmarks WHERE id = ?", (benchmark_id,))
        row = cursor.fetchone()
        if not row:
            print("Error: Could not find the benchmark in the database!")
            return
        
        import json
        emb_id, emb_json = row
        embedding = json.loads(emb_json)
        
        print(f"Stored Benchmark ID: {emb_id}")
        print(f"Embedding type: {type(embedding)}")
        print(f"Embedding vector dimension: {len(embedding)}")
        
        if len(embedding) == 384:
            print("\nSUCCESS: Stored embedding dimension is exactly 384 (MiniLM local model)!")
        else:
            print(f"\nFAILURE: Expected embedding dimension 384, but got {len(embedding)}!")
            
        conn.close()
    except Exception as e:
        print(f"Database verification failed: {e}")

if __name__ == "__main__":
    run_test()
