"""
Course Recommendation FastAPI
File: main.py

FastAPI application for course recommendations using the trained CourseRecommendationSystem.
Provides REST endpoints to get course recommendations for students.
"""

from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Import your recommendation system
from course_recommender import CourseRecommendationSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app instance
app = FastAPI(
    title="Course Recommendation API",
    description="AI-powered course recommendation system for students",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
recommender = None

class StudentProfile(BaseModel):
    """Student profile input model"""
    
    # Basic information
    gender: str = Field(..., description="Gender: M or F")
    stream: str = Field(..., description="Academic stream: Science, Commerce, or Arts")
    
    # Subject scores (0-100)
    physics: Optional[int] = Field(0, ge=0, le=100)
    chemistry: Optional[int] = Field(0, ge=0, le=100)
    mathematics: Optional[int] = Field(0, ge=0, le=100)
    biology: Optional[int] = Field(0, ge=0, le=100)
    computer_science: Optional[int] = Field(0, ge=0, le=100)
    english: Optional[int] = Field(0, ge=0, le=100)
    accountancy: Optional[int] = Field(0, ge=0, le=100)
    business_studies: Optional[int] = Field(0, ge=0, le=100)
    economics: Optional[int] = Field(0, ge=0, le=100)
    banking: Optional[int] = Field(0, ge=0, le=100)
    history: Optional[int] = Field(0, ge=0, le=100)
    political_science: Optional[int] = Field(0, ge=0, le=100)
    sociology: Optional[int] = Field(0, ge=0, le=100)
    education: Optional[int] = Field(0, ge=0, le=100)
    geography: Optional[int] = Field(0, ge=0, le=100)
    hindi: Optional[int] = Field(0, ge=0, le=100)
    
    # Interest levels (0-10)
    interest_physics: Optional[int] = Field(0, ge=0, le=10)
    interest_chemistry: Optional[int] = Field(0, ge=0, le=10)
    interest_mathematics: Optional[int] = Field(0, ge=0, le=10)
    interest_biology: Optional[int] = Field(0, ge=0, le=10)
    interest_computer_science: Optional[int] = Field(0, ge=0, le=10)
    interest_english: Optional[int] = Field(0, ge=0, le=10)
    interest_accountancy: Optional[int] = Field(0, ge=0, le=10)
    interest_business_studies: Optional[int] = Field(0, ge=0, le=10)
    interest_economics: Optional[int] = Field(0, ge=0, le=10)
    interest_banking: Optional[int] = Field(0, ge=0, le=10)
    interest_history: Optional[int] = Field(0, ge=0, le=10)
    interest_political_science: Optional[int] = Field(0, ge=0, le=10)
    interest_sociology: Optional[int] = Field(0, ge=0, le=10)
    interest_education: Optional[int] = Field(0, ge=0, le=10)
    interest_geography: Optional[int] = Field(0, ge=0, le=10)
    interest_hindi: Optional[int] = Field(0, ge=0, le=10)
    
    # Aptitude scores (0-10)
    analytical: Optional[int] = Field(0, ge=0, le=10)
    verbal: Optional[int] = Field(0, ge=0, le=10)
    creativity: Optional[int] = Field(0, ge=0, le=10)
    numerical: Optional[int] = Field(0, ge=0, le=10)
    
    @validator('gender')
    def validate_gender(cls, v):
        if v.upper() not in ['M', 'F']:
            raise ValueError('Gender must be M or F')
        return v.upper()
    
    @validator('stream')
    def validate_stream(cls, v):
        if v.title() not in ['Science', 'Commerce', 'Arts']:
            raise ValueError('Stream must be Science, Commerce, or Arts')
        return v.title()

class CourseRecommendation(BaseModel):
    """Single course recommendation"""
    course: str = Field(..., description="Course name")
    score: float = Field(..., description="Recommendation score")

class RecommendationResponse(BaseModel):
    """API response model for recommendations"""
    success: bool = Field(..., description="Request success status")
    student_info: Dict[str, Any] = Field(..., description="Student profile summary")
    recommendations: List[CourseRecommendation] = Field(..., description="Course recommendations")
    total_courses: int = Field(..., description="Number of recommendations returned")

class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = Field(False, description="Request success status")
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")

def load_model():
    """Load the trained recommendation model"""
    global recommender
    
    try:
        model_path = Path("course_recommendation_model.pkl")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        recommender = CourseRecommendationSystem()
        recommender.load_model(str(model_path))
        logger.info("Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_model()
    if not success:
        logger.warning("API started without model - predictions will fail")

@app.get("/", tags=["Health Check"])
async def root():
    """Root endpoint - API health check"""
    model_status = "loaded" if recommender is not None else "not loaded"
    return {
        "message": "Course Recommendation API",
        "status": "running",
        "model_status": model_status,
        "version": "1.0.0"
    }

@app.get("/health", tags=["Health Check"])
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": recommender is not None,
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.post("/recommend", 
          response_model=RecommendationResponse,
          responses={
              400: {"model": ErrorResponse},
              500: {"model": ErrorResponse}
          },
          tags=["Recommendations"])
async def get_recommendations(
    student: StudentProfile,
    top_k: int = Query(5, ge=1, le=20, description="Number of recommendations to return"),
    filter_by_stream: bool = Query(True, description="Filter courses by student's stream")
):
    """
    Get course recommendations for a student
    
    - **student**: Student profile with scores and interests
    - **top_k**: Number of top recommendations to return (1-20)
    - **filter_by_stream**: Whether to filter by student's academic stream
    """
    
    if recommender is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Recommendation model not loaded"
        )
    
    try:
        # Convert Pydantic model to dictionary
        student_profile = student.dict()
        
        # Get recommendations
        recommendations_df = recommender.predict_for_student(
            student_profile, 
            top_k=top_k, 
            filter_by_stream=filter_by_stream
        )
        
        # Convert to response format
        recommendations = [
            CourseRecommendation(
                course=row['Course'],
                score=float(row['Score'])
            )
            for _, row in recommendations_df.iterrows()
        ]
        
        # Create student info summary
        student_info = {
            "gender": student.gender,
            "stream": student.stream,
            "filter_by_stream": filter_by_stream,
            "key_subjects": get_key_subjects(student_profile),
            "top_interests": get_top_interests(student_profile),
            "aptitude_strengths": get_aptitude_strengths(student_profile)
        }
        
        return RecommendationResponse(
            success=True,
            student_info=student_info,
            recommendations=recommendations,
            total_courses=len(recommendations)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating recommendations: {str(e)}"
        )

@app.post("/recommend/batch",
          tags=["Recommendations"])
async def get_batch_recommendations(
    students: List[StudentProfile],
    top_k: int = Query(3, ge=1, le=10, description="Number of recommendations per student")
):
    """
    Get recommendations for multiple students at once
    
    - **students**: List of student profiles
    - **top_k**: Number of recommendations per student
    """
    
    if recommender is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Recommendation model not loaded"
        )
    
    if len(students) > 50:  # Limit batch size
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 50 students per batch"
        )
    
    try:
        results = []
        
        for i, student in enumerate(students):
            student_profile = student.dict()
            
            recommendations_df = recommender.predict_for_student(
                student_profile, 
                top_k=top_k
            )
            
            recommendations = [
                {"course": row['Course'], "score": float(row['Score'])}
                for _, row in recommendations_df.iterrows()
            ]
            
            results.append({
                "student_index": i,
                "gender": student.gender,
                "stream": student.stream,
                "recommendations": recommendations
            })
        
        return {
            "success": True,
            "total_students": len(students),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing batch: {str(e)}"
        )

@app.get("/courses", tags=["Information"])
async def get_available_courses():
    """Get list of all available courses"""
    
    if recommender is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Recommendation model not loaded"
        )
    
    try:
        courses = recommender.course_le.classes_.tolist()
        
        # Group by stream
        course_streams = {}
        for course in courses:
            stream = recommender.course_prefix(course)
            if stream not in course_streams:
                course_streams[stream] = []
            course_streams[stream].append(course)
        
        return {
            "success": True,
            "total_courses": len(courses),
            "all_courses": courses,
            "by_stream": course_streams
        }
        
    except Exception as e:
        logger.error(f"Error getting courses: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving courses: {str(e)}"
        )

@app.get("/model/info", tags=["Model Information"])
async def get_model_info():
    """Get information about the loaded model"""
    
    if recommender is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Recommendation model not loaded"
        )
    
    try:
        feature_importance = recommender.get_feature_importance(top_n=10)
        
        return {
            "success": True,
            "model_type": "LightGBM Ranker",
            "features_count": len(recommender.feature_cols),
            "student_features_count": len(recommender.student_features),
            "use_subject_penalties": recommender.use_subject_penalties,
            "penalty_strength": recommender.penalty_strength,
            "bonus_strength": recommender.bonus_strength,
            "top_features": feature_importance.head(10).to_dict('records') if feature_importance is not None else []
        }
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving model info: {str(e)}"
        )

# Helper functions
def get_key_subjects(profile: dict) -> list:
    """Get subjects with highest scores"""
    subject_scores = {
        k: v for k, v in profile.items() 
        if not k.startswith('interest_') and k not in ['gender', 'stream', 'analytical', 'verbal', 'creativity', 'numerical'] 
        and v > 0
    }
    
    # Sort by score and return top 3
    sorted_subjects = sorted(subject_scores.items(), key=lambda x: x[1], reverse=True)
    return [{"subject": k, "score": v} for k, v in sorted_subjects[:3]]

def get_top_interests(profile: dict) -> list:
    """Get highest interest subjects"""
    interests = {
        k.replace('interest_', ''): v for k, v in profile.items() 
        if k.startswith('interest_') and v > 0
    }
    
    sorted_interests = sorted(interests.items(), key=lambda x: x[1], reverse=True)
    return [{"subject": k, "interest": v} for k, v in sorted_interests[:3]]

def get_aptitude_strengths(profile: dict) -> list:
    """Get strongest aptitude areas"""
    aptitudes = {
        k: v for k, v in profile.items() 
        if k in ['analytical', 'verbal', 'creativity', 'numerical'] and v > 0
    }
    
    sorted_aptitudes = sorted(aptitudes.items(), key=lambda x: x[1], reverse=True)
    return [{"aptitude": k, "score": v} for k, v in sorted_aptitudes]

# Example usage endpoint
@app.get("/example", tags=["Examples"])
async def get_example_student():
    """Get an example student profile for testing"""
    return {
        "example_student": {
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
        },
        "api_usage": {
            "endpoint": "POST /recommend",
            "description": "Send the example student data to get recommendations"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)