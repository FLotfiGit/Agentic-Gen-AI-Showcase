"""
Example: Multimodal Vision-Language Agent

This example demonstrates how to use the multimodal agent for vision-language
tasks such as image understanding and visual question answering.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.multimodal_agent import MultimodalAgent, ImageInput


def main():
    print("=" * 60)
    print("MULTIMODAL VISION-LANGUAGE AGENT EXAMPLE")
    print("=" * 60)
    print()
    
    # Initialize the agent
    print("Initializing Multimodal Agent...")
    agent = MultimodalAgent()
    print("✅ Agent initialized")
    print()
    
    # Example 1: Image Understanding
    print("Example 1: Image Understanding")
    print("-" * 60)
    
    # Create mock image inputs (in production, these would be actual images)
    image1 = ImageInput(
        image_data=None,  # Would be actual PIL Image or numpy array
        image_id="outdoor_scene_001",
        format="mock",
        metadata={'description': 'Outdoor scene with nature elements'}
    )
    
    print(f"Processing image: {image1.image_id}")
    vision_result = agent.process_image(image1)
    
    print(f"\nDescription: {vision_result.description}")
    print(f"Scene Type: {vision_result.scene_type}")
    print(f"Confidence: {vision_result.confidence:.2%}")
    print(f"\nDetected Objects ({len(vision_result.objects)}):")
    for obj in vision_result.objects:
        print(f"  - {obj['class']:12s} ({obj['confidence']:.2%} confidence)")
        print(f"    BBox: {obj['bbox']}")
    print()
    
    # Example 2: Visual Question Answering
    print("Example 2: Visual Question Answering (VQA)")
    print("-" * 60)
    
    image2 = ImageInput(
        image_data=None,
        image_id="indoor_office_001",
        format="mock",
        metadata={'description': 'Office environment'}
    )
    
    questions = [
        "What do you see in this image?",
        "How many objects are in the scene?",
        "Where was this photo taken?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        result = agent.visual_question_answering(image2, question)
        print(f"A: {result.text_response}")
        print(f"   Confidence: {result.confidence:.2%}")
        print(f"   Reasoning steps:")
        for step in result.reasoning_trace:
            print(f"     - {step}")
    print()
    
    # Example 3: Scene Description
    print("Example 3: Scene Description at Different Detail Levels")
    print("-" * 60)
    
    image3 = ImageInput(
        image_data=None,
        image_id="urban_scene_001",
        format="mock",
        metadata={'description': 'Urban cityscape'}
    )
    
    detail_levels = ['brief', 'medium', 'detailed']
    
    for level in detail_levels:
        print(f"\n{level.upper()} Description:")
        description = agent.describe_scene(image3, detail_level=level)
        print(description)
    print()
    
    # Example 4: Image Comparison
    print("Example 4: Image Comparison")
    print("-" * 60)
    
    image_a = ImageInput(
        image_data=None,
        image_id="scene_a",
        format="mock",
        metadata={'description': 'First scene'}
    )
    
    image_b = ImageInput(
        image_data=None,
        image_id="scene_b",
        format="mock",
        metadata={'description': 'Second scene'}
    )
    
    print(f"Comparing {image_a.image_id} and {image_b.image_id}...")
    comparison = agent.compare_images(image_a, image_b)
    
    print(f"\nImage A: {comparison['image1_description']}")
    print(f"Image B: {comparison['image2_description']}")
    print(f"\nCommon objects: {', '.join(comparison['common_objects']) if comparison['common_objects'] else 'None'}")
    print(f"Unique to Image A: {', '.join(comparison['unique_to_image1']) if comparison['unique_to_image1'] else 'None'}")
    print(f"Unique to Image B: {', '.join(comparison['unique_to_image2']) if comparison['unique_to_image2'] else 'None'}")
    print(f"\nScene Types:")
    print(f"  Image A: {comparison['scene_types']['image1']}")
    print(f"  Image B: {comparison['scene_types']['image2']}")
    print(f"  Similar scenes: {comparison['scene_similarity']}")
    print()
    
    # Example 5: Use Cases
    print("Example 5: Practical Use Cases")
    print("-" * 60)
    print("""
    The Multimodal Vision-Language Agent can be used for:
    
    1. Visual Question Answering (VQA)
       - Answer questions about images
       - Understand context and relationships
    
    2. Image Captioning and Description
       - Generate natural language descriptions
       - Varying levels of detail
    
    3. Object Detection and Recognition
       - Identify objects in scenes
       - Provide bounding boxes and confidence scores
    
    4. Scene Understanding
       - Classify scene types (indoor, outdoor, urban, etc.)
       - Understand spatial relationships
    
    5. Image Comparison
       - Find similarities and differences
       - Content-based comparison
    
    6. Accessibility
       - Describe images for visually impaired users
       - Generate alt-text for web content
    
    7. Content Moderation
       - Analyze image content
       - Detect inappropriate content
    
    8. Visual Search
       - Find similar images
       - Content-based retrieval
    """)
    
    print("=" * 60)
    print("✅ Multimodal Agent examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
