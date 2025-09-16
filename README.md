Course Recommendation API
An AI-powered course recommendation system that provides personalized undergraduate course suggestions based on student academic performance, interests, and aptitudes.
Features
•	Personalized Recommendations: Advanced ML model analyzes student data to suggest optimal courses
•	Multi-Stream Support: Supports Science, Commerce, and Arts academic streams
•	Subject-Course Alignment: Direct matching between student strengths and course requirements
•	Batch Processing: Handle multiple students simultaneously
•	Interactive Documentation: Built-in Swagger/OpenAPI docs
•	Comprehensive Validation: Input validation with detailed error messages
Quick Start
Prerequisites
•	Python 3.8+
•	pip package manager
Installation
1.	Clone the repository:
git clone https://github.com/yourusername/course-recommendation-api.git
cd course-recommendation-api
2.	Install dependencies:
pip install -r requirements.txt
3.	Start the API server:
python main.py
4.	Test the API:
python api_test.py
The API will be available at http://localhost:8000
Docker Setup (Optional)
# Build the image
docker build -t course-recommendation-api .

# Run the container
docker run -p 8000:8000 course-recommendation-api
API Usage
Health Check
curl http://localhost:8000/health
Get Course Recommendations
curl -X POST "http://localhost:8000/recommend?top_k=3" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "M",
    "stream": "Science",
    "mathematics": 92,
    "physics": 85,
    "chemistry": 78,
    "interest_mathematics": 9,
    "interest_physics": 8,
    "analytical": 8,
    "numerical": 9
  }'
Response Format
{
  "success": true,
  "student_info": {
    "gender": "M",
    "stream": "Science",
    "key_subjects": [
      {"subject": "mathematics", "score": 92},
      {"subject": "physics", "score": 85}
    ]
  },
  "recommendations": [
    {"course": "B.Sc. Mathematics", "score": 0.845},
    {"course": "B.Sc. Physics", "score": 0.782},
    {"course": "B.Sc. Computer Science", "score": 0.654}
  ],
  "total_courses": 3
}
Available Endpoints
Method	Endpoint	Description
GET	/	API health check
GET	/health	Detailed health status
POST	/recommend	Get course recommendations for single student
POST	/recommend/batch	Get recommendations for multiple students
GET	/courses	List all available courses
GET	/model/info	Get model information and feature importance
GET	/example	Get sample student data for testing
Student Data Format
Required Fields
•	gender: "M" or "F"
•	stream: "Science", "Commerce", or "Arts"
Subject Scores (0-100)
Science Stream:
•	physics, chemistry, mathematics, biology, computer_science, english
Commerce Stream:
•	english, accountancy, business_studies, economics, banking
Arts Stream:
•	english, history, political_science, sociology, education, geography, hindi
Interest Levels (0-10, Optional)
Rate student interest in each subject (0 = No interest, 10 = Very high interest)
•	interest_physics, interest_chemistry, interest_mathematics, etc.
Aptitude Scores (0-10, Optional)
•	analytical: Problem-solving and logical reasoning
•	verbal: Language and communication skills
•	creativity: Creative and artistic thinking
•	numerical: Mathematical and quantitative skills
Integration Examples
Python
import requests

def get_recommendations(student_data):
    response = requests.post(
        "http://localhost:8000/recommend",
        json=student_data,
        params={"top_k": 5}
    )
    return response.json()

# Example student
student = {
    "gender": "F",
    "stream": "Commerce",
    "economics": 94,
    "banking": 88,
    "interest_economics": 10,
    "analytical": 8
}

result = get_recommendations(student)
for rec in result['recommendations']:
    print(f"{rec['course']}: {rec['score']:.3f}")
JavaScript
async function getRecommendations(studentData) {
    const response = await fetch('http://localhost:8000/recommend?top_k=5', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(studentData)
    });
    return response.json();
}

const student = {
    gender: "M",
    stream: "Science",
    mathematics: 92,
    physics: 85,
    interest_mathematics: 9,
    analytical: 8
};

getRecommendations(student).then(result => {
    result.recommendations.forEach(rec => 
        console.log(`${rec.course}: ${rec.score.toFixed(3)}`)
    );
});
Model Architecture
The system uses a LightGBM Ranker with advanced feature engineering:
•	Subject-Course Alignment: Direct matching between student performance and course requirements
•	Stream-Specific Features: Tailored analysis for Science, Commerce, and Arts streams
•	Performance-Interest Correlation: Combines academic scores with interest levels
•	Aptitude Matching: Aligns student aptitudes with course demands
•	Penalty/Bonus System: Applies penalties for weak subjects and bonuses for strong performance
Key Features Used by Model
1.	Subject performance averages by stream
2.	Interest level correlations
3.	Direct subject-course alignment scores
4.	Aptitude-stream matching
5.	Performance-interest correlation factors
File Structure
course-recommendation-api/
├── main.py                    # FastAPI application
├── course_recommender.py      # ML model implementation
├── api_test.py               # API testing script
├── requirements.txt          # Python dependencies
├── course_recommendation_model.pkl  # Trained model file
├── documentation.pdf         # Complete API documentation
├── LICENSE                   # MIT license
└── README.md                # This file
Testing
Run the comprehensive test suite:
python api_test.py
This will test:
•	API connectivity
•	Model loading
•	Single student recommendations
•	Different stream types (Science, Commerce, Arts)
•	Error handling
•	All endpoints
API Documentation
•	Interactive Docs: http://localhost:8000/docs
•	ReDoc: http://localhost:8000/redoc
•	Complete PDF Guide: See documentation.pdf
Error Handling
The API provides detailed error responses:
{
  "success": false,
  "error": "Validation error",
  "detail": "Gender must be M or F"
}
Common HTTP status codes:
•	200: Success
•	400: Bad Request (invalid data)
•	422: Validation Error
•	500: Internal Server Error
Performance
•	Prediction Speed: <100ms per student
•	Batch Processing: Up to 50 students per request
•	Memory Usage: ~200MB with model loaded
•	Accuracy: 85%+ NDCG@3 on validation data
Deployment
Production Considerations
1.	Environment Variables:
export API_HOST=0.0.0.0
export API_PORT=8000
export MODEL_PATH=./course_recommendation_model.pkl
2.	Process Management:
# Using gunicorn
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
3.	Docker Deployment:
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["python", "main.py"]
Cloud Deployment
The API can be deployed on:
•	AWS: EC2, ECS, Lambda (with adaptations)
•	Google Cloud: Cloud Run, Compute Engine
•	Azure: Container Instances, App Service
•	Heroku: Direct deployment with Procfile
Model Training
To retrain the model with new data:
from course_recommender import CourseRecommendationSystem

# Load your training data
# df should have columns: student_id, gender, stream, subject scores, interests, candidate_course, relevance

recommender = CourseRecommendationSystem()
recommender.train(df)
recommender.save_model('new_model.pkl')
Contributing
1.	Fork the repository
2.	Create a feature branch: git checkout -b feature-name
3.	Make changes and test thoroughly
4.	Submit a pull request
Development Setup
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
python api_test.py

# Format code
black *.py

# Lint code
flake8 *.py
License
This project is licensed under the MIT License - see the LICENSE file for details.
Support
•	Issues: Report bugs or request features via GitHub Issues
•	Documentation: Complete API guide in documentation.pdf
•	Health Check: Monitor API status at /health
Sample Data
Science Student
{
  "gender": "M",
  "stream": "Science",
  "physics": 92, "chemistry": 78, "mathematics": 95,
  "biology": 65, "computer_science": 82, "english": 75,
  "interest_mathematics": 10, "interest_physics": 9,
  "analytical": 9, "numerical": 10
}
Commerce Student
{
  "gender": "F",
  "stream": "Commerce",
  "economics": 94, "banking": 91, "accountancy": 78,
  "business_studies": 85, "english": 82,
  "interest_economics": 10, "interest_banking": 9,
  "analytical": 8, "numerical": 9
}
Arts Student
{
  "gender": "F",
  "stream": "Arts",
  "english": 88, "history": 82, "political_science": 85,
  "sociology": 78, "geography": 75,
  "interest_english": 9, "interest_history": 8,
  "verbal": 10, "creativity": 9
}
________________________________________
Built with Python, FastAPI, and LightGBM
For technical questions or integration support, please refer to the comprehensive documentation or create an issue.

