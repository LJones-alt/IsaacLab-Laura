#!/usr/bin/env python3
"""
Script to find and copy the best validation model checkpoint for each model
in the ensemble training runs.

The script searches through each model directory in Dev-IK-Rel-Place-v0,
finds the checkpoint with the lowest validation loss, and copies it to a
centralized 'best_models' folder.
"""

import os
import re
import shutil
from pathlib import Path
from typing import Optional, Tuple


def parse_validation_loss(filename: str) -> Optional[float]:
    """
    Extract validation loss from checkpoint filename.
    
    Expected format: model_epoch_XXX_best_validation_YYY.YYY.pth
    
    Args:
        filename: The checkpoint filename
        
    Returns:
        The validation loss as a float, or None if parsing fails
    """
    pattern = r'model_epoch_\d+_best_validation_([\d.]+)\.pth'
    match = re.search(pattern, filename)
    if match:
        return float(match.group(1))
    return None


def find_best_checkpoint(model_dir: Path) -> Optional[Tuple[Path, float]]:
    """
    Find the checkpoint with the lowest validation loss in a model directory.
    
    Args:
        model_dir: Path to the model directory (e.g., model0, model1, etc.)
        
    Returns:
        Tuple of (checkpoint_path, validation_loss) or None if no checkpoints found
    """
    # Search for all checkpoint files in the models subdirectory
    checkpoint_pattern = "model_epoch_*_best_validation_*.pth"
    checkpoints = list(model_dir.rglob(checkpoint_pattern))
    
    if not checkpoints:
        print(f"  ⚠️  No checkpoints found in {model_dir.name}")
        return None
    
    best_checkpoint = None
    best_loss = float('inf')
    
    for checkpoint in checkpoints:
        loss = parse_validation_loss(checkpoint.name)
        if loss is not None and loss < best_loss:
            best_loss = loss
            best_checkpoint = checkpoint
    
    if best_checkpoint:
        return (best_checkpoint, best_loss)
    
    print(f"  ⚠️  Could not parse validation loss from checkpoints in {model_dir.name}")
    return None


def main():
    """Main function to process all model directories."""
    # Define paths - handle both host and Docker container paths
    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir / "insert/Dev-IK-Rel-Insert-v0"
    output_dir = base_dir / "best_models"
    
    # Create output directory (including parent directories if needed)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}\n")
    
    # Find all model directories (model0, model1, ..., modelN)
    model_dirs = sorted([d for d in base_dir.iterdir() 
                        if d.is_dir() and d.name.startswith("model") 
                        and d.name != "best_models"])
    
    if not model_dirs:
        print("❌ No model directories found!")
        return
    
    print(f"Found {len(model_dirs)} model directories\n")
    print("=" * 80)
    
    # Process each model directory
    results = []
    for model_dir in model_dirs:
        print(f"\n🔍 Processing {model_dir.name}...")
        
        result = find_best_checkpoint(model_dir)
        if result:
            checkpoint_path, validation_loss = result
            print(f"  ✅ Best checkpoint: {checkpoint_path.name}")
            print(f"  📊 Validation loss: {validation_loss}")
            
            # Copy to output directory with model name prefix
            output_filename = f"{model_dir.name}_{checkpoint_path.name}"
            output_path = output_dir / output_filename
            
            shutil.copy2(checkpoint_path, output_path)
            print(f"  📋 Copied to: {output_filename}")
            
            results.append({
                'model': model_dir.name,
                'checkpoint': checkpoint_path.name,
                'loss': validation_loss,
                'output': output_filename
            })
    
    # Print summary
    print("\n" + "=" * 80)
    print("\nSUMMARY")
    print("=" * 80)
    print(f"\nProcessed {len(model_dirs)} models")
    print(f"Successfully copied {len(results)} best checkpoints\n")
    
    if results:
        print("Best checkpoints by validation loss:")
        print("-" * 80)
        sorted_results = sorted(results, key=lambda x: x['loss'])
        for i, r in enumerate(sorted_results, 1):
            print(f"{i:2d}. {r['model']:8s} | Loss: {r['loss']:12.2f} | {r['checkpoint']}")
        
        # Save file paths to text file
        save_paths_to_file(output_dir, results)
        
        print(f"\nAll best models saved to: {output_dir}")
    else:
        print("No checkpoints were copied")


def save_paths_to_file(output_dir: Path, results: list) -> None:
    """
    Save all copied checkpoint file paths to a text file.
    
    Args:
        output_dir: Directory where the checkpoints were copied
        results: List of dictionaries containing model information
    """
    txt_file = output_dir / "best_model_paths.txt"
    
    with open(txt_file, 'w') as f:
        for result in sorted(results, key=lambda x: x['model']):
            file_path = output_dir / result['output']
            f.write(f"{file_path}\n")
    
    print(f"📝 File paths saved to: {txt_file}")


if __name__ == "__main__":
    main()
