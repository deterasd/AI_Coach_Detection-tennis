import os
import json
import re
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, make_response
import glob

# Import validation functions
from modules.step1_reprojection_error import validate_reprojection_analysis
from modules.step2_bone_consistency import validate_bone_consistency_analysis
from modules.step3_depth_reasonableness import validate_depth_reasonableness_analysis

app = Flask(__name__, static_folder='templates')

# Configuration
# DATA_DIR is now dynamic based on request, but we can keep a default
DEFAULT_DATA_DIR = 'data'
CAMERA_CONFIG_FILE = 'camera_configs.json'

@app.route('/')
def index():
    return send_from_directory('templates', 'dashboard.html')

@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory('data', filename)

@app.route('/<path:path>')
def serve_static(path):
    # Check if file exists in templates folder
    if os.path.exists(os.path.join('templates', path)):
        return send_from_directory('templates', path)
    # Otherwise serve from root directory (for data files, etc.)
    return send_from_directory('.', path)

@app.route('/api/folders', methods=['GET'])
def list_folders():
    """List all subdirectories recursively that contain JSON files."""
    folders = []
    try:
        # Walk through the directory tree
        for root, dirs, files in os.walk('.'):
            # Modify dirs in-place to skip unwanted directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and not d.startswith('__') and d not in ['templates', 'modules', 'static', 'venv', 'env', '.git']]
            
            # Check if current directory contains any .json files
            if any(f.lower().endswith('.json') for f in files):
                # Get relative path
                rel_path = os.path.relpath(root, '.').replace('\\', '/')
                if rel_path == '.':
                    continue
                folders.append(rel_path)
                
        return jsonify({"folders": sorted(folders)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/files', methods=['GET'])
def list_files():
    """List all JSON files in the specified directory."""
    folder = request.args.get('folder', DEFAULT_DATA_DIR)
    
    if not os.path.exists(folder):
        return jsonify({"error": f"Directory {folder} not found"}), 404
    
    files = glob.glob(os.path.join(folder, '*.json'))
    # Return relative paths
    files = [f.replace('\\', '/') for f in files]
    return jsonify({"files": files})

@app.route('/api/validate/step1', methods=['POST'])
def validate_step1():
    data = request.json
    try:
        json_3d = data.get('json_3d')
        json_2d_side = data.get('json_2d_side')
        json_2d_45 = data.get('json_2d_45')
        p1_data = data.get('p1')
        p2_data = data.get('p2')

        if not all([json_3d, json_2d_side, json_2d_45, p1_data, p2_data]):
            return jsonify({"error": "Missing required parameters"}), 400

        # Convert matrices to numpy arrays
        P1 = np.array(p1_data)
        P2 = np.array(p2_data)

        # Run validation
        # The function returns the results dict, but also saves to a file.
        # We want the output file path to pass to the frontend.
        # We can let the function generate the default path, or specify one.
        # Let's specify one to be sure.
        
        # Create results directory
        output_dir = os.path.join(os.path.dirname(json_3d), 'Verification Result')
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(json_3d))[0]
        output_filename = f"{base_name}_step1_reprojection_error_results.json"
        output_path = os.path.join(output_dir, output_filename)
        
        validate_reprojection_analysis(
            json_3d,
            json_2d_side,
            json_2d_45,
            P1, P2,
            output_json_path=output_path
        )
        
        return jsonify({
            "status": "success", 
            "result_file": output_path.replace('\\', '/')
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/validate/step2', methods=['POST'])
def validate_step2():
    data = request.json
    try:
        json_3d = data.get('json_3d')
        
        if not json_3d:
            return jsonify({"error": "Missing required parameters"}), 400

        # Create results directory
        output_dir = os.path.join(os.path.dirname(json_3d), 'Verification Result')
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(json_3d))[0]
        output_filename = f"{base_name}_step2_bone_consistency_results.json"
        output_path = os.path.join(output_dir, output_filename)

        validate_bone_consistency_analysis(
            json_3d,
            output_json_path=output_path,
            config_path=None
        )

        return jsonify({
            "status": "success", 
            "result_file": output_path.replace('\\', '/')
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/validate/step3', methods=['POST'])
def validate_step3():
    data = request.json
    try:
        json_3d = data.get('json_3d')

        if not json_3d:
            return jsonify({"error": "Missing required parameters"}), 400

        output_dir = os.path.join(os.path.dirname(json_3d), 'Verification Result')
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(json_3d))[0]
        output_filename = f"{base_name}_step3_depth_reasonableness_results.json"
        output_path = os.path.join(output_dir, output_filename)

        validate_depth_reasonableness_analysis(
            json_3d,
            output_json_path=output_path,
            config_path=None
        )

        return jsonify({
            "status": "success",
            "result_file": output_path.replace('\\', '/')
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def load_camera_configs():
    if not os.path.exists(CAMERA_CONFIG_FILE):
        return {}
    try:
        with open(CAMERA_CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}

def save_camera_configs(configs):
    # Dump with default indentation
    json_str = json.dumps(configs, indent=4)
    
    # Post-process to collapse inner lists of numbers (matrices rows)
    # Match arrays containing only numbers, whitespace, commas, dots, minus, e/E
    # This regex avoids matching arrays that contain objects or other arrays (no [ or ])
    pattern = r'\[\s*([0-9.,\-\seE]+)\s*\]'
    
    def replacer(match):
        content = match.group(1)
        # Split by comma, strip whitespace, filter empty, join back
        items = [item.strip() for item in content.split(',') if item.strip()]
        return "[" + ", ".join(items) + "]"
        
    new_json_str = re.sub(pattern, replacer, json_str)

    with open(CAMERA_CONFIG_FILE, 'w', encoding='utf-8') as f:
        f.write(new_json_str)

@app.route('/api/camera-settings', methods=['GET', 'POST'])
def handle_camera_settings():
    if request.method == 'GET':
        configs = load_camera_configs()
        return jsonify(configs)
    
    elif request.method == 'POST':
        data = request.json
        name = data.get('name')
        p1 = data.get('p1')
        p2 = data.get('p2')
        
        if not name or not p1 or not p2:
            return jsonify({"error": "Missing name, p1, or p2"}), 400
            
        configs = load_camera_configs()
        configs[name] = {
            "p1": p1,
            "p2": p2
        }
        save_camera_configs(configs)
        return jsonify({"status": "success", "message": f"Configuration '{name}' saved."})

@app.route('/api/camera-settings/<name>', methods=['DELETE'])
def delete_camera_setting(name):
    configs = load_camera_configs()
    if name in configs:
        del configs[name]
        save_camera_configs(configs)
        return jsonify({"status": "success", "message": f"Configuration '{name}' deleted."})
    return jsonify({"error": "Configuration not found"}), 404

@app.route('/api/export/offline-report', methods=['GET'])
def export_offline_report():
    report_type = request.args.get('type')
    result_file = request.args.get('file')
    save_to_disk = request.args.get('save_to_disk', 'false').lower() == 'true'
    
    if not report_type or not result_file:
        return jsonify({"error": "Missing parameters"}), 400
        
    if not os.path.exists(result_file):
        return jsonify({"error": "Result file not found"}), 404
        
    # Determine template file
    if report_type == 'step1':
        template_file = 'templates/step1_reprojection_error.html'
    elif report_type == 'step2':
        template_file = 'templates/step2_bone_consistency.html'
    elif report_type == 'step3':
        template_file = 'templates/step3_depth_reasonableness.html'
    else:
        return jsonify({"error": "Invalid report type"}), 400
        
    if not os.path.exists(template_file):
        return jsonify({"error": "Template file not found"}), 404
        
    try:
        # Read JSON data
        with open(result_file, 'r', encoding='utf-8') as f:
            json_content = f.read()
            
        # Read HTML template
        with open(template_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
            
        # Inject data
        # We inject it into the head so it's available before any body scripts run
        injection_script = f"""
    <script>
        window.OFFLINE_DATA = {json_content};
        console.log("Offline data loaded successfully");
    </script>
"""
        
        if '</head>' in html_content:
            modified_html = html_content.replace('</head>', f'{injection_script}\n</head>')
        else:
            # Fallback if no head tag
            modified_html = injection_script + html_content
        
        # If save_to_disk is requested, save the file next to the result json
        if save_to_disk:
            output_dir = os.path.dirname(result_file)
            base_name = os.path.splitext(os.path.basename(result_file))[0]
            # Remove _results suffix if present to avoid redundancy
            base_name = base_name.replace('_results', '')
            html_filename = f"{base_name}_report.html"
            output_path = os.path.join(output_dir, html_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(modified_html)
                
            return jsonify({
                "status": "success",
                "message": f"Report saved to {output_path}",
                "file_path": output_path.replace('\\', '/')
            })

        # Create response for download
        response = make_response(modified_html)
        response.headers['Content-Disposition'] = f'attachment; filename={report_type}_offline_report.html'
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        
        return response
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting server on http://localhost:5000")
    app.run(debug=True, port=5000)
