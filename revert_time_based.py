import os
import json
import shutil
import glob

def relax_strict_json_configs(target_dir="configs/pending_hypotheses"):
    print("=========================================")
    print("   🤖 QUANT AUTO-RELAXATION ENGINE       ")
    print("=========================================")
    
    # Grab all JSON configs in the pending folder
    json_files = glob.glob(os.path.join(target_dir, "*.json"))
    if not json_files:
        print(f"❌ No JSON files found in {target_dir}")
        return

    relaxed_count = 0
    
    for filepath in json_files:
        with open(filepath, 'r') as f:
            try:
                config = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Skipping {os.path.basename(filepath)} - Invalid JSON format.")
                continue
        
        # Check if the logic contains strict secondary filters
        logic = config.get('logic', {})
        filters = logic.get('filters', [])
        
        if len(filters) > 0:
            file_name = os.path.basename(filepath)
            print(f"🔧 Relaxing strict filters in: {file_name}")
            
            # 1. Safely backup the strict version
            backup_path = filepath + ".strict_bak"
            shutil.copy(filepath, backup_path)
            
            # 2. STRIP THE SECONDARY FILTERS (Leave the core triggers intact)
            config['logic']['filters'] = []
            
            # 3. Mark the metadata so it shows in the terminal
            if 'metadata' in config:
                current_name = config['metadata'].get('name', '')
                if "(RELAXED)" not in current_name:
                    config['metadata']['name'] = current_name + " (RELAXED)"
                
            # 4. Save the relaxed version back
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=4)
            
            relaxed_count += 1
            
    print("\n=========================================")
    print(f"✅ Protocol Complete. Relaxed {relaxed_count} hypotheses.")
    print("📂 Original strict configs were saved with '.strict_bak' extensions.")
    print("▶️ You can now run 'python batch_runner.py' to test the core triggers.")
    print("=========================================")

if __name__ == "__main__":
    relax_strict_json_configs()