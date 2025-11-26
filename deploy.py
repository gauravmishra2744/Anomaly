from waitress import serve
from app import app

if __name__ == '__main__':
    print("="*70)
    print("ANOMALY DETECTION SYSTEM - PRODUCTION SERVER")
    print("="*70)
    print("\nServer starting on http://0.0.0.0:8080")
    print("Access locally: http://localhost:8080")
    print("Press Ctrl+C to stop\n")
    print("="*70)
    
    serve(app, host='0.0.0.0', port=8080, threads=4)
