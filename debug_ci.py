import os
import sys
from pathlib import Path

# Add the docker/utils directory to path
sys.path.append(os.path.join(os.getcwd(), 'docker/utils'))
from container_interface import ContainerInterface

def debug_ci():
    ci = ContainerInterface(
        context_dir=Path('docker'),
        profile='ros2'
    )
    
    # Mock X11 check since it needs user input/state
    ci.add_yamls += ["--file", "x11.yaml"]
    
    print("add_yamls:", ci.add_yamls)
    print("add_profiles:", ci.add_profiles)
    print("add_env_files:", ci.add_env_files)
    
    cmd = ["docker", "compose"] + ci.add_yamls + ci.add_profiles + ci.add_env_files + ["up", "--detach", "--build", "--remove-orphans"]
    print("Full command:", ' '.join(cmd))

if __name__ == "__main__":
    debug_ci()
