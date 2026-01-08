#!/usr/bin/env python3
"""
Basic usage example for ccRLM.

This demonstrates how to use the RLM abstraction to process
a complex prompt using the recursive agent approach.

Prerequisites:
    - ANTHROPIC_API_KEY environment variable set
    - Claude Code CLI installed: npm install -g @anthropic-ai/claude-code
    - Package installed: pip install -e .

Usage:
    python examples/basic_usage.py
"""

import os
from pathlib import Path

from ccrlm import RLM, RLMConfig


def simple_example():
    """Simplest possible usage with a string prompt."""
    print("=== Simple Example ===\n")

    rlm = RLM()
    result = rlm.invoke("""
    Analyze the following data and provide insights:

    Sales by region:
    - North: $1.2M (+15% YoY)
    - South: $0.8M (-5% YoY)
    - East: $1.5M (+22% YoY)
    - West: $0.9M (+3% YoY)

    Key questions:
    1. Which region is performing best?
    2. What might explain the South's decline?
    3. What should be the priority for next quarter?
    """)

    print(f"Success: {result.success}")
    print(f"Answer:\n{result.answer}")
    print(f"\nMetrics: {result.metrics}")


def messages_example():
    """Using messages format like the Anthropic API."""
    print("\n=== Messages Example ===\n")

    rlm = RLM()

    # Multi-turn conversation
    result = rlm.invoke(messages=[
        {
            "role": "user",
            "content": "I have a dataset of customer reviews. Here's a sample:\n\n"
                       "1. 'Great product, fast shipping!' - 5 stars\n"
                       "2. 'Broke after a week, terrible quality' - 1 star\n"
                       "3. 'Exactly what I needed, good value' - 4 stars\n"
                       "4. 'Customer service was unhelpful' - 2 stars\n"
                       "5. 'Best purchase I've made this year!' - 5 stars"
        },
        {
            "role": "assistant",
            "content": "I can see your customer review dataset. It contains 5 reviews "
                       "with ratings ranging from 1 to 5 stars. What would you like me to analyze?"
        },
        {
            "role": "user",
            "content": "Identify the main themes and suggest improvements based on the negative feedback."
        },
    ])

    print(f"Success: {result.success}")
    print(f"Answer:\n{result.answer}")


def system_prompt_example():
    """Using a system prompt with messages."""
    print("\n=== System Prompt Example ===\n")

    rlm = RLM()

    result = rlm.invoke(
        messages=[
            {"role": "user", "content": "Review this Python code:\n\n"
                                        "def get_user(id):\n"
                                        "    query = f'SELECT * FROM users WHERE id = {id}'\n"
                                        "    return db.execute(query)"}
        ],
        system="You are a security expert specializing in code review. "
               "Identify vulnerabilities and suggest fixes with code examples."
    )

    print(f"Success: {result.success}")
    print(f"Answer:\n{result.answer}")


def configured_example():
    """Example with custom configuration."""
    print("\n=== Configured Example ===\n")

    config = RLMConfig(
        # Subcall settings
        max_tokens_per_subcall=800,
        max_depth=3,
        max_scripts=10,
        submodel="claude-haiku-4-20250414",

        # Root agent settings
        root_model="claude-sonnet-4-20250514",
        max_turns=30,

        # Keep workspace for inspection
        preserve_workspace=True,
    )

    rlm = RLM(config=config)
    result = rlm.invoke("What is 2 + 2? Explain your reasoning.")

    print(f"Success: {result.success}")
    print(f"Answer:\n{result.answer}")

    if result.workspace_path:
        print(f"\nWorkspace preserved at: {result.workspace_path}")
        print("You can inspect the files there.")


def streaming_example():
    """Example with message streaming."""
    print("\n=== Streaming Example ===\n")

    def on_message(msg):
        """Called for each agent message."""
        msg_type = msg.get("type", "unknown")
        if msg_type == "assistant":
            content = msg.get("message", {}).get("content", [])
            for block in content:
                if block.get("type") == "text":
                    print(f"Agent: {block.get('text', '')[:100]}...")
        elif msg_type == "result":
            print(f"Tool result received")

    rlm = RLM(on_message=on_message)
    result = rlm.invoke("List 3 interesting facts about Python programming.")

    print(f"\nFinal answer:\n{result.answer}")


def large_prompt_example():
    """Example that would benefit from chunking."""
    print("\n=== Large Prompt Example ===\n")

    # Generate a large prompt
    large_content = "\n\n".join([
        f"## Section {i}\n\nThis is section {i} with some content. " * 20
        for i in range(50)
    ])

    prompt = f"""
    Analyze the following document and provide a summary of key themes:

    {large_content}

    Provide:
    1. Main themes identified
    2. Key points per section (grouped)
    3. Overall summary
    """

    print(f"Prompt size: {len(prompt):,} characters")

    config = RLMConfig(
        max_tokens_per_subcall=1000,
        preserve_workspace=True,
    )

    rlm = RLM(config=config)
    result = rlm.invoke(prompt)

    print(f"Success: {result.success}")
    print(f"Evidence items collected: {len(result.evidence)}")
    print(f"Errors: {len(result.errors)}")

    if result.answer:
        print(f"\nAnswer preview:\n{result.answer[:500]}...")


def one_shot_example():
    """Using the convenience function."""
    print("\n=== One-Shot Example ===\n")

    from ccrlm.rlm import invoke

    result = invoke(
        prompt="Explain recursion in programming with an example.",
        max_tokens=500,
        model="claude-sonnet-4-20250514",
    )

    print(f"Success: {result.success}")
    print(f"Answer:\n{result.answer}")


if __name__ == "__main__":
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY=your_key_here")
        exit(1)

    # Run examples
    try:
        simple_example()
    except Exception as e:
        print(f"Simple example failed: {e}")

    # Uncomment to run other examples:
    # messages_example()        # Multi-turn conversation
    # system_prompt_example()   # With system prompt
    # configured_example()
    # streaming_example()
    # large_prompt_example()
    # one_shot_example()
