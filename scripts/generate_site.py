import os
import datetime
import subprocess
from pathlib import Path

def run_command(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout

def main():
    print("Generating demo site...")
    
    # Run setup check
    setup_output = run_command("python mousiki_cli.py setup")
    
    # Run recommendation
    recs_output = run_command("python mousiki_cli.py recommend --user-id 1")
    
    # Read template
    template_path = Path("public/template.html")
    with open(template_path, "r") as f:
        template = f.read()
    
    # Replace placeholders
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    html = template.replace("{{ SETUP_OUTPUT }}", setup_output)
    html = html.replace("{{ RECS_OUTPUT }}", recs_output)
    html = html.replace("{{ TIMESTAMP }}", timestamp)
    
    # Write output
    output_path = Path("public/index.html")
    with open(output_path, "w") as f:
        f.write(html)
        
    print(f"Site generated at {output_path}")

if __name__ == "__main__":
    main()
