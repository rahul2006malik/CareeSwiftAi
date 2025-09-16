"""
Course Recommendation System - Main Model
File: course_recommender.py

A hybrid LightGBM-based course recommendation system with subject-course alignment penalties.
Addresses the Education ranking problem through enhanced feature engineering.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ndcg_score
import warnings
warnings.filterwarnings('ignore')

class CourseRecommendationSystem:
    def __init__(self, use_subject_penalties=True, penalty_strength=0.6, bonus_strength=0.8):
        """
        Initialize the Course Recommendation System
        
        Args:
            use_subject_penalties (bool): Whether to apply penalties for low subject scores
            penalty_strength (float): Penalty for courses where student scores <30
            bonus_strength (float): Bonus for courses where student scores >80
        """
        self.ranker = None
        self.gender_le = LabelEncoder()
        self.stream_le = LabelEncoder()
        self.course_le = LabelEncoder()
        self.coursestream_le = LabelEncoder()
        self.feature_cols = None
        self.student_features = None
        
        # Subject-course matching parameters (fixes Education problem)
        self.use_subject_penalties = use_subject_penalties
        self.penalty_strength = penalty_strength
        self.bonus_strength = bonus_strength
        
        # Subject-course mapping for direct matching
        self.subject_course_mapping = {
            'physics': ['physics'],
            'chemistry': ['chemistry'],
            'mathematics': ['mathematics', 'math'],
            'biology': ['biology', 'botany', 'zoology'],
            'computer_science': ['computer science', 'computer'],
            'english': ['english'],
            'accountancy': ['accounting', 'accountancy'],
            'business_studies': ['business'],
            'economics': ['economics'],
            'banking': ['banking'],
            'history': ['history'],
            'political_science': ['political science'],
            'sociology': ['sociology'],
            'education': ['education'],
            'geography': ['geography'],
            'hindi': ['hindi']
        }
    
    def course_prefix(self, name):
        """Extract course stream from course name"""
        if pd.isna(name): 
            return "UNK"
        s = str(name).upper()
        if s.startswith("B.A") or s.startswith("BA "): 
            return "Arts"
        if "B.SC" in s or s.startswith("BSC") or s.startswith("BCA") or s.startswith("B.TECH") or s.startswith("B.E"): 
            return "Science"
        if "B.COM" in s or s.startswith("BCOM") or s.startswith("BBA"): 
            return "Commerce"
        return "Other"
    
    def calculate_subject_course_alignment(self, student_scores, course_name):
        """Calculate direct subject-course alignment with penalties/bonuses"""
        course_lower = course_name.lower()
        alignment_score = 0
        
        for subject, keywords in self.subject_course_mapping.items():
            if any(keyword in course_lower for keyword in keywords):
                student_score = student_scores.get(subject, 0)
                
                if self.use_subject_penalties:
                    if student_score >= 80:  # High performance bonus
                        alignment_score += self.bonus_strength
                    elif student_score <= 30:  # Low performance penalty
                        alignment_score -= self.penalty_strength
                    else:  # Neutral performance
                        alignment_score += (student_score / 100) * 0.2
                else:
                    alignment_score += (student_score / 100) * 0.3
                
                # Add interest alignment
                interest_score = student_scores.get(f'interest_{subject}', 0)
                alignment_score += (interest_score / 10) * 0.2
                
                break  # Only apply to first matching subject
        
        return alignment_score
    
    def create_enhanced_features(self, df):
        """Create enhanced features combining all approaches"""
        data = df.copy()
        
        # Course stream derivation
        data["course_stream"] = data["candidate_course"].apply(self.course_prefix)
        
        # Subject groupings
        science_subjects = ['physics', 'chemistry', 'mathematics', 'biology', 'computer_science']
        commerce_subjects = ['accountancy', 'business_studies', 'economics', 'banking']
        arts_subjects = ['english', 'history', 'political_science', 'sociology', 'education', 'geography', 'hindi']
        
        # Stream performance averages
        data['science_avg'] = data[science_subjects].mean(axis=1)
        data['commerce_avg'] = data[commerce_subjects].mean(axis=1) 
        data['arts_avg'] = data[arts_subjects].mean(axis=1)
        data['overall_academic_avg'] = data[science_subjects + commerce_subjects + arts_subjects].mean(axis=1)
        
        # Interest averages
        science_interests = ['interest_' + s for s in science_subjects]
        commerce_interests = ['interest_' + s for s in commerce_subjects]
        arts_interests = ['interest_' + s for s in arts_subjects]
        
        data['science_interest_avg'] = data[science_interests].mean(axis=1)
        data['commerce_interest_avg'] = data[commerce_interests].mean(axis=1)
        data['arts_interest_avg'] = data[arts_interests].mean(axis=1)
        data['overall_interest_avg'] = data[science_interests + commerce_interests + arts_interests].mean(axis=1)
        
        # Stream-course alignment
        data['stream_course_match'] = (
            ((data['stream'] == 'Science') & (data['course_stream'] == 'Science')) |
            ((data['stream'] == 'Commerce') & (data['course_stream'] == 'Commerce')) |
            ((data['stream'] == 'Arts') & (data['course_stream'] == 'Arts'))
        ).astype(int)
        
        # CRITICAL: Direct subject-course alignment (fixes Education problem)
        data['subject_course_alignment'] = 0
        for idx, row in data.iterrows():
            student_scores = {col: row[col] for col in science_subjects + commerce_subjects + arts_subjects}
            student_scores.update({f'interest_{col}': row[f'interest_{col}'] for col in science_subjects + commerce_subjects + arts_subjects})
            
            alignment = self.calculate_subject_course_alignment(student_scores, row['candidate_course'])
            data.at[idx, 'subject_course_alignment'] = alignment
        
        # Aptitude features
        aptitude_cols = ['analytical', 'verbal', 'creativity', 'numerical']
        data['aptitude_total'] = data[aptitude_cols].sum(axis=1)
        
        # Stream-specific aptitude matching
        data['stream_aptitude_match'] = 0
        data.loc[data['stream'] == 'Science', 'stream_aptitude_match'] = (
            data.loc[data['stream'] == 'Science', 'analytical'] + 
            data.loc[data['stream'] == 'Science', 'numerical']
        ) / 2
        data.loc[data['stream'] == 'Commerce', 'stream_aptitude_match'] = (
            data.loc[data['stream'] == 'Commerce', 'numerical'] + 
            data.loc[data['stream'] == 'Commerce', 'analytical']
        ) / 2
        data.loc[data['stream'] == 'Arts', 'stream_aptitude_match'] = (
            data.loc[data['stream'] == 'Arts', 'verbal'] + 
            data.loc[data['stream'] == 'Arts', 'creativity']
        ) / 2
        
        # Performance-Interest correlation per stream
        data['science_perf_interest_corr'] = (data['science_avg'] / 100) * (data['science_interest_avg'] / 10)
        data['commerce_perf_interest_corr'] = (data['commerce_avg'] / 100) * (data['commerce_interest_avg'] / 10)
        data['arts_perf_interest_corr'] = (data['arts_avg'] / 100) * (data['arts_interest_avg'] / 10)
        
        return data
    
    def prepare_features(self, df):
        """Prepare all features for training"""
        data = self.create_enhanced_features(df)
        
        # Encodings
        data["gender_enc"] = self.gender_le.fit_transform(data["gender"].astype(str))
        data["student_stream_enc"] = self.stream_le.fit_transform(data["stream"].astype(str))
        data["candidate_course_idx"] = self.course_le.fit_transform(data["candidate_course"].astype(str))
        data["course_stream_idx"] = self.coursestream_le.fit_transform(data["course_stream"].astype(str))
        
        # Student features
        self.student_features = [c for c in [
            'physics','chemistry','mathematics','biology','computer_science','english',
            'accountancy','business_studies','economics','banking','history',
            'political_science','sociology','education','geography','hindi',
            'interest_physics','interest_chemistry','interest_mathematics','interest_biology',
            'interest_computer_science','interest_english','interest_accountancy',
            'interest_business_studies','interest_economics','interest_banking',
            'interest_history','interest_political_science','interest_sociology',
            'interest_education','interest_geography','interest_hindi',
            'analytical','verbal','creativity','numerical'
        ] if c in data.columns]
        
        # Enhanced features
        enhanced_features = [
            'science_avg', 'commerce_avg', 'arts_avg', 'overall_academic_avg',
            'science_interest_avg', 'commerce_interest_avg', 'arts_interest_avg', 'overall_interest_avg',
            'stream_course_match', 'subject_course_alignment',
            'aptitude_total', 'stream_aptitude_match',
            'science_perf_interest_corr', 'commerce_perf_interest_corr', 'arts_perf_interest_corr'
        ]
        
        self.feature_cols = (self.student_features + enhanced_features + 
                            ["gender_enc", "student_stream_enc", "candidate_course_idx", "course_stream_idx"])
        
        return data
    
    def train(self, df, test_size=0.2, random_state=42):
        """Train the model"""
        print("Preparing enhanced features with subject-course penalties...")
        data = self.prepare_features(df)
        
        print("Splitting data by student_id...")
        student_ids = data["student_id"].unique()
        train_ids, test_ids = train_test_split(student_ids, test_size=test_size, random_state=random_state)
        
        train_df = data[data["student_id"].isin(train_ids)].reset_index(drop=True)
        test_df = data[data["student_id"].isin(test_ids)].reset_index(drop=True)
        
        X_train, y_train = train_df[self.feature_cols], train_df["relevance"].astype(int)
        X_test, y_test = test_df[self.feature_cols], test_df["relevance"].astype(int)
        
        groups_train = train_df.groupby("student_id").size().to_numpy()
        groups_test = test_df.groupby("student_id").size().to_numpy()
        
        print(f"Training: {len(train_df)} samples, {len(groups_train)} groups")
        print(f"Test: {len(test_df)} samples, {len(groups_test)} groups")
        
        # Optimized parameters
        self.ranker = lgb.LGBMRanker(
            objective="lambdarank", 
            metric="ndcg", 
            num_leaves=63,
            learning_rate=0.06,
            n_estimators=500,
            feature_fraction=0.85,
            bagging_fraction=0.85,
            bagging_freq=7,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=random_state,
            importance_type='gain',
            force_col_wise=True,
            verbosity=-1
        )
        
        print("Training LGBMRanker...")
        self.ranker.fit(
            X_train, y_train,
            group=groups_train,
            eval_set=[(X_test, y_test)],
            eval_group=[groups_test],
            eval_at=[3],
            callbacks=[lgb.early_stopping(stopping_rounds=40), lgb.log_evaluation(period=50)]
        )
        
        return self.ranker
    
    def predict_for_student(self, student_profile, top_k=3, filter_by_stream=True):
        """Predict top courses for a student"""
        if self.ranker is None:
            raise ValueError("Model not trained yet!")
        
        # Fill missing features
        for c in self.student_features:
            if c not in student_profile:
                student_profile[c] = 0
        
        # Encode student categories
        try:
            gender_enc = self.gender_le.transform([student_profile.get("gender", "M")])[0]
        except:
            gender_enc = 0
            
        try:
            student_stream_enc = self.stream_le.transform([student_profile.get("stream", "Science")])[0]  
        except:
            student_stream_enc = 0
        
        # Get candidates
        all_courses = pd.DataFrame({
            'candidate_course': self.course_le.classes_,
            'course_stream': [self.course_prefix(c) for c in self.course_le.classes_]
        })
        
        if filter_by_stream and "stream" in student_profile:
            target_stream = student_profile["stream"]
            candidates = all_courses[all_courses["course_stream"] == target_stream]
            if candidates.empty:
                candidates = all_courses
        else:
            candidates = all_courses
        
        # Build prediction rows with all enhanced features
        rows = []
        
        for _, r in candidates.iterrows():
            rec = {k: student_profile[k] for k in self.student_features}
            
            # Calculate all enhanced features
            science_subjects = ['physics', 'chemistry', 'mathematics', 'biology', 'computer_science']
            commerce_subjects = ['accountancy', 'business_studies', 'economics', 'banking']
            arts_subjects = ['english', 'history', 'political_science', 'sociology', 'education', 'geography', 'hindi']
            
            rec['science_avg'] = np.mean([student_profile.get(s, 0) for s in science_subjects])
            rec['commerce_avg'] = np.mean([student_profile.get(s, 0) for s in commerce_subjects])
            rec['arts_avg'] = np.mean([student_profile.get(s, 0) for s in arts_subjects])
            rec['overall_academic_avg'] = np.mean([student_profile.get(s, 0) for s in science_subjects + commerce_subjects + arts_subjects])
            
            rec['science_interest_avg'] = np.mean([student_profile.get(f'interest_{s}', 0) for s in science_subjects])
            rec['commerce_interest_avg'] = np.mean([student_profile.get(f'interest_{s}', 0) for s in commerce_subjects])
            rec['arts_interest_avg'] = np.mean([student_profile.get(f'interest_{s}', 0) for s in arts_subjects])
            rec['overall_interest_avg'] = np.mean([student_profile.get(f'interest_{s}', 0) for s in science_subjects + commerce_subjects + arts_subjects])
            
            # Stream-course match
            student_stream = student_profile.get("stream", "Science")
            course_stream = r["course_stream"]
            rec['stream_course_match'] = 1 if student_stream == course_stream else 0
            
            # Subject-course alignment
            rec['subject_course_alignment'] = self.calculate_subject_course_alignment(student_profile, r["candidate_course"])
            
            # Aptitude features
            aptitude_cols = ['analytical', 'verbal', 'creativity', 'numerical']
            rec['aptitude_total'] = sum([student_profile.get(c, 0) for c in aptitude_cols])
            
            # Stream-aptitude match
            if student_stream == 'Science':
                rec['stream_aptitude_match'] = (student_profile.get('analytical', 0) + student_profile.get('numerical', 0)) / 2
            elif student_stream == 'Commerce':
                rec['stream_aptitude_match'] = (student_profile.get('numerical', 0) + student_profile.get('analytical', 0)) / 2
            else:  # Arts
                rec['stream_aptitude_match'] = (student_profile.get('verbal', 0) + student_profile.get('creativity', 0)) / 2
            
            # Performance-Interest correlations
            rec['science_perf_interest_corr'] = (rec['science_avg'] / 100) * (rec['science_interest_avg'] / 10)
            rec['commerce_perf_interest_corr'] = (rec['commerce_avg'] / 100) * (rec['commerce_interest_avg'] / 10)
            rec['arts_perf_interest_corr'] = (rec['arts_avg'] / 100) * (rec['arts_interest_avg'] / 10)
            
            # Encoded features
            rec["gender_enc"] = gender_enc
            rec["student_stream_enc"] = student_stream_enc
            rec["candidate_course_idx"] = self.course_le.transform([r["candidate_course"]])[0]
            rec["course_stream_idx"] = self.coursestream_le.transform([r["course_stream"]])[0]
            rec["candidate_course_name"] = r["candidate_course"]
            
            rows.append(rec)
        
        # Predict and rank
        pred_df = pd.DataFrame(rows)
        scores = self.ranker.predict(pred_df[self.feature_cols])
        pred_df["score"] = scores
        
        # Get top-k recommendations
        top_courses = pred_df.sort_values("score", ascending=False).head(top_k)
        
        result = top_courses[["candidate_course_name", "score"]].reset_index(drop=True)
        result.columns = ["Course", "Score"]
        
        return result
    
    def get_feature_importance(self, top_n=15):
        """Get feature importance"""
        if self.ranker is None:
            print("Model not trained yet!")
            return None
            
        importance = self.ranker.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"Top {top_n} Most Important Features:")
        print("-" * 60)
        for _, row in feature_imp.head(top_n).iterrows():
            print(f"{row['feature']:.<45} {row['importance']:>10.1f}")
        
        return feature_imp
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.ranker is None:
            raise ValueError("No trained model to save!")
        
        import joblib
        model_data = {
            'ranker': self.ranker,
            'gender_le': self.gender_le,
            'stream_le': self.stream_le,
            'course_le': self.course_le,
            'coursestream_le': self.coursestream_le,
            'feature_cols': self.feature_cols,
            'student_features': self.student_features,
            'use_subject_penalties': self.use_subject_penalties,
            'penalty_strength': self.penalty_strength,
            'bonus_strength': self.bonus_strength,
            'subject_course_mapping': self.subject_course_mapping
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        import joblib
        model_data = joblib.load(filepath)
        
        self.ranker = model_data['ranker']
        self.gender_le = model_data['gender_le']
        self.stream_le = model_data['stream_le']
        self.course_le = model_data['course_le']
        self.coursestream_le = model_data['coursestream_le']
        self.feature_cols = model_data['feature_cols']
        self.student_features = model_data['student_features']
        self.use_subject_penalties = model_data['use_subject_penalties']
        self.penalty_strength = model_data['penalty_strength']
        self.bonus_strength = model_data['bonus_strength']
        self.subject_course_mapping = model_data['subject_course_mapping']
        
        print(f"Model loaded from {filepath}")