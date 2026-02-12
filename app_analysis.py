import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
from pathlib import Path
import json
import traceback

# Import the analysis function
# Ensure analyze_from_file.py is in the same directory or PYTHONPATH
try:
    from analyze_from_file import run_analysis
except ImportError:
    print("Module analyze_from_file not found. Please ensure it is in the same directory.")
    run_analysis = None

app = FastAPI(title="Tennis Analysis Server")

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = Path("temp_upload")
TEMP_DIR.mkdir(exist_ok=True)

@app.post("/analyze")
async def analyze_trajectory(file: UploadFile = File(...)):
    if not run_analysis:
        raise HTTPException(status_code=500, detail="Analysis module not loaded")
    
    try:
        # 1. Save uploaded file
        filename = file.filename
        file_path = TEMP_DIR / filename
        
        # --- ADAPTER for Raw Files ---
        # If user uploads a raw trajectory file, we MUST rename it to match 
        # the strict requirement of analyze_from_file.py: "(3D_trajectory_smoothed).json"
        if filename.endswith("(3D_trajectory).json"):
            new_filename = filename.replace("(3D_trajectory).json", "(3D_trajectory_smoothed).json")
            file_path = TEMP_DIR / new_filename
            print(f"Renaming raw file {filename} -> {new_filename} to satisfy script requirements")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"File saved at: {file_path}")
        
        # 2. Run Analysis
        # Note: run_analysis expects suffix (3D_trajectory_smoothed).json
        
        try:
            # Run the analysis (this might raise ValueError if suffix is wrong)
            success = run_analysis(
                trajectory_path=str(file_path),
                knn_dataset_path="knn_dataset_new.json",
                n_neighbors=3
            )
        except ValueError as e:
            # Catch the specific suffix error from analyze_from_file.py
            shutil.rmtree(TEMP_DIR, ignore_errors=True)
            return JSONResponse(
                status_code=400,
                content={"error": str(e), "detail": "File Type Error. Please upload a 3D trajectory file ending in '(3D_trajectory_smoothed).json'."}
            )
        except Exception as e:
            shutil.rmtree(TEMP_DIR, ignore_errors=True)
            print(f"Analysis Failed: {e}")
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content={"error": str(e), "detail": "Internal Server Error during analysis."}
            )

        if not success:
            return JSONResponse(status_code=500, content={"error": "Analysis returned False status"})
            
        # 3. Retrieve Results
        # Expected outputs:
        # - {base}_integrated_analysis.json
        # - {base}_gpt_feedback.json
        
        # We need to reconstruct 'base' similar to analyze_from_file._base_and_dir
        # But analyze_from_file handles suffix stripping.
        # Let's search TEMP_DIR for the newest json files or standard names
        
        base_name = filename
        if base_name.endswith("(3D_trajectory_smoothed).json"):
            base_name = base_name[:-len("(3D_trajectory_smoothed).json")]
        elif base_name.endswith("(3D_trajectory).json"):
            base_name = base_name[:-len("(3D_trajectory).json")]
        
        integrated_path = TEMP_DIR / f"{base_name}_integrated_analysis.json"
        gpt_path = TEMP_DIR / f"{base_name}_gpt_feedback.json"
        
        result_data = {
            "integrated": {},
            "gpt_feedback": {}
        }
        
        if integrated_path.exists():
            with open(integrated_path, "r", encoding="utf-8") as f:
                result_data["integrated"] = json.load(f)
        else:
            print(f"Warning: Integrated analysis file not found: {integrated_path}")

        if gpt_path.exists():
            with open(gpt_path, "r", encoding="utf-8") as f:
                result_data["gpt_feedback"] = json.load(f)
        else:
            print(f"Warning: GPT feedback file not found: {gpt_path}")

        return JSONResponse(content=result_data)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Ensure knn_dataset exists
    if not Path("knn_dataset_new.json").exists():
        print("WARNING: knn_dataset_new.json not found in current directory!")
    
    print("Starting server at http://0.0.0.0:8001")
    uvicorn.run("app_analysis:app", host="0.0.0.0", port=8001, reload=True)
