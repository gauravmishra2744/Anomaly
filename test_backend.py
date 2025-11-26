from app import app
import json

print("Testing backend...")

with app.test_client() as client:
    # Test dashboard
    resp = client.get('/api/dashboard')
    print(f"\n1. Dashboard API: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json
        print(f"   Total samples: {data['total_samples']}")
        print(f"   Anomalies: {data['anomalies_detected']}")
        print(f"   Accuracy: {data['accuracy']:.2%}")
    
    # Test prediction
    resp = client.post('/api/predict', 
                       data=json.dumps({'sample_idx': 0}),
                       content_type='application/json')
    print(f"\n2. Predict API: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json
        print(f"   Prediction: {data['prediction']}")
        print(f"   Error: {data['error']:.6f}")
    
    # Test statistics
    resp = client.get('/api/statistics')
    print(f"\n3. Statistics API: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json
        print(f"   Accuracy: {data['accuracy']:.2%}")
        print(f"   Precision: {data['precision']:.2%}")

print("\nBackend test complete!")
