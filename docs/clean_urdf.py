import xml.etree.ElementTree as ET

# Load the original URDF
input_urdf = "franka_robotiq/urdf/panda_robotiq.urdf"
output_urdf = "franka_robotiq/urdf/lula_franka_isaaclab.urdf"

tree = ET.parse(input_urdf)
robot = tree.getroot()

# Default inertial parameters
DEFAULT_MASS = 1.0
DEFAULT_INERTIA = {"ixx": 0.01, "iyy": 0.01, "izz": 0.01, "ixy": 0.0, "ixz": 0.0, "iyz": 0.0}

for link in robot.findall("link"):
    # Add inertial if missing
    if link.find("inertial") is None:
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=str(DEFAULT_MASS))
        inertia_attrib = {k: str(v) for k, v in DEFAULT_INERTIA.items()}
        ET.SubElement(inertial, "inertia", **inertia_attrib)
    
    # Add simple box collision if missing
    if link.find("collision") is None:
        collision = ET.SubElement(link, "collision")
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "box", size="0.05 0.05 0.05")

# Write the cleaned URDF
tree.write(output_urdf, encoding="utf-8", xml_declaration=True)

print(f"Cleaned URDF written to {output_urdf}")