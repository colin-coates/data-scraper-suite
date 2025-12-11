# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
CLI Interface for MJ Data Scraper Suite

Command-line interface for managing and running the scraper suite.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

from .scraper_engine import ScraperEngine, EngineConfig
from .core.plugin_manager import PluginManager


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('scraper.log')
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except ImportError:
        print("PyYAML not installed. Install with: pip install PyYAML")
        return {}
    except Exception as e:
        print(f"Error loading config {config_path}: {e}")
        return {}


async def run_scraper_job(args: argparse.Namespace) -> None:
    """Run a single scraper job."""
    print("🕷️ MJ Data Scraper Suite - Job Runner")
    print("=" * 40)

    # Load configuration
    config_data = {}
    if args.config:
        config_data = load_config(args.config)

    # Create engine config
    engine_config = EngineConfig(
        max_concurrent_jobs=config_data.get('max_concurrent_jobs', 5),
        azure_service_bus_connection=config_data.get('azure_service_bus_connection', ''),
        azure_blob_connection=config_data.get('azure_blob_connection', ''),
        enable_anti_detection=config_data.get('enable_anti_detection', True)
    )

    # Initialize engine
    engine = ScraperEngine(engine_config)
    await engine.initialize()

    # Load plugins
    plugin_manager = PluginManager()
    plugin_manager.load_all_plugins()

    # Register scrapers from plugins
    for plugin_name in plugin_manager.get_loaded_plugins():
        scraper_class = plugin_manager.get_scraper_class(plugin_name)
        if scraper_class:
            engine.register_scraper(plugin_name, scraper_class)

    print(f"Registered {len(engine.scrapers)} scrapers")

    # Create job
    job_data = {
        "scraper_type": args.scraper_type,
        "target": {
            "url": args.target_url
        },
        "priority": args.priority,
        "metadata": {
            "source": "cli",
            "user": args.user or "anonymous"
        }
    }

    # Dispatch job
    try:
        job_id = await engine.dispatch_job(job_data)
        print(f"Job dispatched: {job_id}")

        # Start engine
        await engine.start()

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await engine.cleanup()


async def run_interactive_mode(args: argparse.Namespace) -> None:
    """Run in interactive mode."""
    print("🕷️ MJ Data Scraper Suite - Interactive Mode")
    print("=" * 40)

    # Load configuration
    config_data = {}
    if args.config:
        config_data = load_config(args.config)

    # Create engine
    engine_config = EngineConfig(
        max_concurrent_jobs=config_data.get('max_concurrent_jobs', 5),
        enable_anti_detection=config_data.get('enable_anti_detection', True)
    )

    engine = ScraperEngine(engine_config)
    await engine.initialize()

    # Load plugins
    plugin_manager = PluginManager()
    plugin_manager.load_all_plugins()

    # Register scrapers
    for plugin_name in plugin_manager.get_loaded_plugins():
        scraper_class = plugin_manager.get_scraper_class(plugin_name)
        if scraper_class:
            engine.register_scraper(plugin_name, scraper_class)

    print(f"Available scrapers: {', '.join(engine.scrapers.keys())}")
    print("Type 'help' for commands, 'quit' to exit")

    try:
        while True:
            cmd = input("scraper> ").strip()

            if cmd == 'quit':
                break
            elif cmd == 'help':
                print_commands()
            elif cmd.startswith('scrape '):
                await handle_scrape_command(cmd, engine)
            elif cmd == 'status':
                show_status(engine)
            elif cmd == 'metrics':
                show_metrics(engine)
            else:
                print(f"Unknown command: {cmd}")

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await engine.cleanup()


def print_commands() -> None:
    """Print available commands."""
    print("""
Available commands:
  scrape <type> <url>     - Scrape a URL with specified scraper type
  status                  - Show current engine status
  metrics                 - Show performance metrics
  help                    - Show this help
  quit                    - Exit interactive mode

Examples:
  scrape linkedin https://linkedin.com/in/john-doe
  scrape web https://example.com
""")


async def handle_scrape_command(cmd: str, engine: ScraperEngine) -> None:
    """Handle scrape command in interactive mode."""
    try:
        parts = cmd.split()
        if len(parts) != 3:
            print("Usage: scrape <type> <url>")
            return

        scraper_type, url = parts[1], parts[2]

        job_data = {
            "scraper_type": scraper_type,
            "target": {"url": url},
            "priority": "normal"
        }

        job_id = await engine.dispatch_job(job_data)
        print(f"Job dispatched: {job_id}")

        # In a real implementation, you'd wait for completion or poll status

    except Exception as e:
        print(f"Error dispatching job: {e}")


def show_status(engine: ScraperEngine) -> None:
    """Show engine status."""
    metrics = engine.get_metrics()

    print("Engine Status:")
    print(f"  Running: True")
    print(f"  Registered Scrapers: {len(metrics['registered_scrapers'])}")
    print(f"  Active Scrapers: {len(metrics['active_scrapers'])}")
    print(f"  Queued Jobs: {metrics['queued_jobs']}")
    print(f"  Active Jobs: {metrics['active_jobs']}")
    print(f"  Completed Jobs: {metrics['completed_jobs']}")


def show_metrics(engine: ScraperEngine) -> None:
    """Show engine metrics."""
    metrics = engine.get_metrics()

    print("Performance Metrics:")
    print(f"  Jobs Dispatched: {metrics['jobs_dispatched']}")
    print(f"  Jobs Completed: {metrics['jobs_completed']}")
    print(f"  Jobs Failed: {metrics['jobs_failed']}")
    print(f"  Success Rate: {metrics['success_rate']:.2%}")
    print(f"  Avg Execution Time: {metrics['average_processing_time']:.2f}s")
    print(f"  Uptime: {metrics['uptime_seconds']:.0f}s")


def create_config_template(args: argparse.Namespace) -> None:
    """Create a configuration template."""
    template = {
        "max_concurrent_jobs": 5,
        "enable_anti_detection": True,
        "azure_service_bus_connection": "your-service-bus-connection-string",
        "azure_blob_connection": "your-blob-storage-connection-string",
        "azure_queue_name": "scraping-jobs",
        "azure_blob_container": "scraping-results",
        "default_rate_limit": 1.0,
        "scrapers": {
            "linkedin": {
                "enabled": True,
                "rate_limit": 10,
                "max_profiles_per_session": 50
            },
            "web": {
                "enabled": True,
                "rate_limit": 30,
                "extract_metadata": True,
                "extract_links": True
            }
        }
    }

    output_path = args.output or "config/scraper_config.yaml"

    try:
        import yaml
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(template, f, default_flow_style=False, indent=2)

        print(f"Configuration template created: {output_path}")

    except ImportError:
        # Fallback to JSON
        output_path = output_path.replace('.yaml', '.json')
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(template, f, indent=2)

        print(f"Configuration template created: {output_path}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MJ Data Scraper Suite - Enterprise Web Scraping Platform"
    )

    parser.add_argument(
        "--config", "-c",
        help="Configuration file path"
    )

    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Job runner subcommand
    job_parser = subparsers.add_parser("job", help="Run a single scraping job")
    job_parser.add_argument("scraper_type", help="Type of scraper to use")
    job_parser.add_argument("target_url", help="URL to scrape")
    job_parser.add_argument("--priority", choices=["low", "normal", "high", "urgent"],
                          default="normal", help="Job priority")
    job_parser.add_argument("--user", help="User identifier for the job")

    # Interactive mode subcommand
    interactive_parser = subparsers.add_parser("interactive", help="Run in interactive mode")

    # Config template subcommand
    config_parser = subparsers.add_parser("create-config", help="Create configuration template")
    config_parser.add_argument("--output", "-o", help="Output file path")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Execute command
    if args.command == "job":
        asyncio.run(run_scraper_job(args))
    elif args.command == "interactive":
        asyncio.run(run_interactive_mode(args))
    elif args.command == "create-config":
        create_config_template(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
