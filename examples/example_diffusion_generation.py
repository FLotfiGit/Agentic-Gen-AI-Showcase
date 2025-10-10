"""
Example: Diffusion-Based Generative Model

This example demonstrates how to use diffusion models for image generation,
including text-to-image, image-to-image, and various generation settings.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.diffusion_generation import (
    DiffusionModel,
    DiffusionPipeline,
    GenerationConfig,
    SamplerType
)


def main():
    print("=" * 60)
    print("DIFFUSION-BASED GENERATIVE MODEL EXAMPLE")
    print("=" * 60)
    print()
    
    # Example 1: Basic Text-to-Image Generation
    print("Example 1: Text-to-Image Generation")
    print("-" * 60)
    
    model = DiffusionModel()
    
    config = GenerationConfig(
        prompt="A serene lake surrounded by mountains at sunset, photorealistic",
        negative_prompt="blurry, distorted, low quality",
        width=512,
        height=512,
        num_inference_steps=30,
        guidance_scale=7.5,
        seed=42
    )
    
    print(f"Prompt: '{config.prompt}'")
    print(f"Negative Prompt: '{config.negative_prompt}'")
    print(f"Settings: {config.width}x{config.height}, {config.num_inference_steps} steps")
    print()
    
    result = model.generate(config)
    
    print(f"✅ Generation completed!")
    print(f"   Time taken: {result.generation_time:.2f}s")
    print(f"   Steps: {result.steps_taken}")
    print(f"   Image shape: {result.image_data.shape}")
    print()
    
    # Example 2: Using the Pipeline for Multiple Images
    print("Example 2: Generating Multiple Images")
    print("-" * 60)
    
    pipeline = DiffusionPipeline(model)
    
    prompt = "A futuristic cyberpunk city with neon lights"
    print(f"Generating 3 variations of: '{prompt}'")
    print()
    
    results = pipeline.text_to_image(
        prompt=prompt,
        negative_prompt="anime, cartoon",
        num_images=3,
        num_inference_steps=20,
        seed=100
    )
    
    print(f"✅ Generated {len(results)} images:")
    for i, result in enumerate(results):
        print(f"   Image {i+1}: {result.image_data.shape}, "
              f"time: {result.generation_time:.2f}s")
    print()
    
    # Example 3: Different Quality Presets
    print("Example 3: Quality Presets")
    print("-" * 60)
    
    quality_levels = ['fast', 'balanced', 'high']
    
    for quality in quality_levels:
        print(f"\n{quality.upper()} Quality:")
        config_dict = pipeline.get_optimal_config(quality)
        for key, value in config_dict.items():
            print(f"  {key}: {value}")
    print()
    
    # Example 4: Different Samplers
    print("Example 4: Different Sampling Methods")
    print("-" * 60)
    
    samplers = [
        (SamplerType.DDPM, "Denoising Diffusion Probabilistic Models"),
        (SamplerType.DDIM, "Denoising Diffusion Implicit Models"),
        (SamplerType.EULER, "Euler Method"),
    ]
    
    print("Available sampling methods:")
    for sampler, description in samplers:
        print(f"  - {sampler.value}: {description}")
    print()
    
    # Example 5: Creative Prompts Showcase
    print("Example 5: Creative Prompt Examples")
    print("-" * 60)
    
    creative_prompts = [
        {
            'prompt': 'A magical forest with glowing mushrooms and fireflies at night',
            'style': 'Fantasy',
            'suggested_settings': {'guidance_scale': 8.0, 'steps': 40}
        },
        {
            'prompt': 'Abstract geometric patterns in vibrant colors, digital art',
            'style': 'Abstract',
            'suggested_settings': {'guidance_scale': 7.0, 'steps': 30}
        },
        {
            'prompt': 'A steampunk robot in a Victorian workshop, detailed, 8k',
            'style': 'Steampunk',
            'suggested_settings': {'guidance_scale': 7.5, 'steps': 50}
        },
        {
            'prompt': 'Underwater scene with coral reefs and tropical fish, bright colors',
            'style': 'Nature',
            'suggested_settings': {'guidance_scale': 7.5, 'steps': 35}
        },
        {
            'prompt': 'A cozy library with books and warm lighting, photorealistic',
            'style': 'Interior',
            'suggested_settings': {'guidance_scale': 7.0, 'steps': 40}
        }
    ]
    
    print("Creative prompt suggestions with recommended settings:\n")
    for i, example in enumerate(creative_prompts, 1):
        print(f"{i}. Style: {example['style']}")
        print(f"   Prompt: \"{example['prompt']}\"")
        print(f"   Suggested settings:")
        for key, value in example['suggested_settings'].items():
            print(f"     - {key}: {value}")
        print()
    
    # Example 6: Advanced Techniques
    print("Example 6: Advanced Generation Techniques")
    print("-" * 60)
    print("""
    The diffusion model supports various advanced techniques:
    
    1. Text-to-Image
       - Generate images from text descriptions
       - Use guidance scale to control adherence to prompt
       
    2. Image-to-Image
       - Transform existing images based on prompts
       - Control transformation strength (0.0 to 1.0)
       
    3. Inpainting
       - Fill in masked regions of images
       - Seamlessly blend with surrounding content
       
    4. Outpainting
       - Extend images beyond their borders
       - Generate content that matches the style
       
    5. Negative Prompts
       - Specify what to avoid in generation
       - Improve quality by excluding unwanted elements
       
    6. Guidance Scale
       - Controls how closely to follow the prompt
       - Higher values = stronger prompt adherence
       - Typical range: 7.0 to 9.0
       
    7. Sampling Steps
       - More steps = higher quality but slower
       - Fast: 20-30 steps
       - Balanced: 40-50 steps
       - High quality: 80-100 steps
       
    8. Seeds
       - Use seeds for reproducible results
       - Same seed + same prompt = same image
    """)
    
    # Example 7: Performance Considerations
    print("\nExample 7: Performance Optimization Tips")
    print("-" * 60)
    print("""
    Tips for optimizing generation:
    
    1. Resolution
       - Start with 512x512 for faster iteration
       - Use 768x768 or 1024x1024 for final results
       
    2. Batch Generation
       - Generate multiple variations at once
       - More efficient than sequential generation
       
    3. Sampling Steps
       - Use fewer steps (20-30) during experimentation
       - Increase steps (50-100) for final outputs
       
    4. Guidance Scale
       - 7.5 is a good default
       - Adjust based on results (7.0-9.0 range)
       
    5. Hardware Acceleration
       - Use GPU for significant speedup
       - Consider mixed precision (FP16) for faster inference
       
    6. Model Selection
       - Smaller models for faster generation
       - Larger models for better quality
    """)
    
    print()
    print("=" * 60)
    print("✅ Diffusion Model examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
