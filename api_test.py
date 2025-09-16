"""
API Testing Script
File: test_api.py

Test script to verify the Course Recommendation API is working correctly.
"""

import requests
import json
import time
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - is the server running?")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_detailed_health():
    """Test the detailed health endpoint"""
    print("\nTesting detailed health...")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        data = response.json()
        print(f"Status: {response.status_code}")
        print(f"Model loaded: {data.get('model_loaded', False)}")
        return response.status_code == 200 and data.get('model_loaded', False)
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_single_recommendation():
    """Test single student recommendation"""
    print("\nTesting single recommendation...")
    
    # Science student example
    student = {
        "gender": "M",
        "stream": "Science",
        "physics": 85,
        "chemistry": 78,
        "mathematics": 92,
        "biology": 70,
        "computer_science": 80,
        "english": 75,
        "interest_physics": 8,
        "interest_chemistry": 6,
        "interest_mathematics": 9,
        "interest_biology": 5,
        "interest_computer_science": 7,
        "interest_english": 6,
        "analytical": 8,
        "verbal": 6,
        "creativity": 7,
        "numerical": 9
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/recommend",
            json=student,
            params={"top_k": 3}
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data['success']}")
            print(f"Student: {data['student_info']['gender']} {data['student_info']['stream']}")
            print("Recommendations:")
            
            for i, rec in enumerate(data['recommendations'], 1):
                print(f"  {i}. {rec['course']} (Score: {rec['score']:.3f})")
            
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_commerce_student():
    """Test commerce student recommendation"""
    print("\nTesting commerce student...")
    
    student = {
        "gender": "F",
        "stream": "Commerce",
        "english": 82,
        "accountancy": 78,
        "business_studies": 85,
        "economics": 94,
        "banking": 91,
        "interest_accountancy": 7,
        "interest_business_studies": 8,
        "interest_economics": 10,
        "interest_banking": 9,
        "interest_english": 7,
        "analytical": 8,
        "verbal": 8,
        "creativity": 6,
        "numerical": 9
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/recommend",
            json=student,
            params={"top_k": 5}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("Commerce Recommendations:")
            for i, rec in enumerate(data['recommendations'], 1):
                print(f"  {i}. {rec['course']} (Score: {rec['score']:.3f})")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_arts_student():
    """Test arts student recommendation"""
    print("\nTesting arts student...")
    
    student = {
        "gender": "F",
        "stream": "Arts",
        "english": 88,
        "history": 82,
        "political_science": 79,
        "sociology": 85,
        "education": 75,
        "geography": 80,
        "hindi": 77,
        "interest_english": 9,
        "interest_history": 8,
        "interest_political_science": 7,
        "interest_sociology": 9,
        "interest_education": 6,
        "interest_geography": 7,
        "interest_hindi": 5,
        "analytical": 7,
        "verbal": 9,
        "creativity": 8,
        "numerical": 5
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/recommend",
            json=student,
            params={"top_k": 4}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("Arts Recommendations:")
            for i, rec in enumerate(data['recommendations'], 1):
                print(f"  {i}. {rec['course']} (Score: {rec['score']:.3f})")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_get_courses():
    """Test get available courses endpoint"""
    print("\nTesting get courses...")
    
    try:
        response = requests.get(f"{BASE_URL}/courses")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Total courses available: {data['total_courses']}")
            print("Courses by stream:")
            for stream, courses in data['by_stream'].items():
                print(f"  {stream}: {len(courses)} courses")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_example_endpoint():
    """Test the example endpoint"""
    print("\nTesting example endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/example")
        
        if response.status_code == 200:
            data = response.json()
            print("Example student data retrieved successfully")
            print(f"Stream: {data['example_student']['stream']}")
            print(f"Gender: {data['example_student']['gender']}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_basic_tests():
    """Run essential tests to verify API is working"""
    print("=== Course Recommendation API Basic Tests ===\n")
    
    # Test 1: Health Check
    health_ok = test_health_check()
    if not health_ok:
        print("\n❌ API is not running or not accessible. Please:")
        print("1. Make sure you've started the API server: python main.py")
        print("2. Check if the server is running on http://localhost:8000")
        return False
    
    # Test 2: Detailed Health
    model_ok = test_detailed_health()
    if not model_ok:
        print("\n❌ Model is not loaded. Please check:")
        print("1. course_recommendation_model.pkl exists in your directory")
        print("2. course_recommender.py is properly implemented")
        print("3. All required dependencies are installed")
        return False
    
    # Test 3: Basic recommendation
    rec_ok = test_single_recommendation()
    if not rec_ok:
        print("\n❌ Recommendation endpoint failed")
        return False
    
    print("\n✅ All basic tests passed! API is working correctly.")
    return True

def run_all_tests():
    """Run comprehensive test suite"""
    print("=== Course Recommendation API Complete Test Suite ===\n")
    
    tests = [
        ("Health Check", test_health_check),
        ("Detailed Health", test_detailed_health),
        ("Single Recommendation (Science)", test_single_recommendation),
        ("Commerce Student", test_commerce_student),
        ("Arts Student", test_arts_student),
        ("Get Courses", test_get_courses),
        ("Example Endpoint", test_example_endpoint),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
        
        time.sleep(0.5)  # Brief pause between tests
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<35} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    return passed == total

def main():
    """Main function to run tests"""
    print("Course Recommendation API Test Script")
    print("====================================")
    
    # Check if we can connect to the API first
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        print(f"✅ API server is accessible at {BASE_URL}")
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to API at {BASE_URL}")
        print("\nPlease make sure:")
        print("1. The API server is running (python main.py)")
        print("2. The server is accessible at http://localhost:8000")
        print("3. No firewall is blocking the connection")
        return
    except Exception as e:
        print(f"❌ Error connecting to API: {e}")
        return
    
    # Ask user which tests to run
    print("\nChoose test mode:")
    print("1. Basic tests (recommended for first run)")
    print("2. Complete test suite")
    print("3. Individual test selection")
    
    choice = input("\nEnter your choice (1-3) or press Enter for basic tests: ").strip()
    
    if choice == "2":
        run_all_tests()
    elif choice == "3":
        print("\nAvailable individual tests:")
        print("1. Health check")
        print("2. Science student recommendation")
        print("3. Commerce student recommendation")
        print("4. Arts student recommendation")
        print("5. Get all courses")
        
        test_choice = input("Enter test number: ").strip()
        
        if test_choice == "1":
            test_health_check()
            test_detailed_health()
        elif test_choice == "2":
            test_single_recommendation()
        elif test_choice == "3":
            test_commerce_student()
        elif test_choice == "4":
            test_arts_student()
        elif test_choice == "5":
            test_get_courses()
        else:
            print("Invalid choice")
    else:
        # Default to basic tests
        run_basic_tests()

if __name__ == "__main__":
    main()