# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Plugin Manager for MJ Data Scraper Suite

Provides plugin architecture for dynamically loading and managing scraper plugins.
Supports hot-reloading, dependency management, and plugin lifecycle.
"""

import importlib
import inspect
import logging
import os
import sys
from typing import Dict, Any, List, Optional, Type, Callable
from pathlib import Path
from dataclasses import dataclass, field

from .base_scraper import BaseScraper, ScraperConfig

logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """Information about a loaded plugin."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    dependencies: List[str] = field(default_factory=list)
    scraper_class: Optional[Type[BaseScraper]] = None
    config_class: Optional[Type[ScraperConfig]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    loaded: bool = False
    enabled: bool = True


class PluginManager:
    """
    Manages scraper plugins with dynamic loading and lifecycle management.
    """

    def __init__(self, plugin_dirs: Optional[List[str]] = None):
        self.plugin_dirs = plugin_dirs or [
            "scrapers",
            "plugins",
            "mj_data_scraper_suite/plugins"
        ]

        self.plugins: Dict[str, PluginInfo] = {}
        self.loaded_plugins: Dict[str, Any] = {}

        # Plugin lifecycle callbacks
        self.on_plugin_loaded: Optional[Callable[[str, PluginInfo], None]] = None
        self.on_plugin_unloaded: Optional[Callable[[str], None]] = None
        self.on_plugin_error: Optional[Callable[[str, Exception], None]] = None

        logger.info("PluginManager initialized")

    def discover_plugins(self) -> List[str]:
        """
        Discover available plugins in configured directories.

        Returns:
            List of discovered plugin names
        """
        discovered = []

        for plugin_dir in self.plugin_dirs:
            plugin_path = Path(plugin_dir)

            if not plugin_path.exists():
                continue

            # Look for Python files
            for py_file in plugin_path.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue

                plugin_name = py_file.stem
                discovered.append(plugin_name)

                # Also check subdirectories
                if py_file.is_dir():
                    for sub_py in py_file.glob("*.py"):
                        if not sub_py.name.startswith("_"):
                            discovered.append(f"{plugin_name}.{sub_py.stem}")

        # Remove duplicates while preserving order
        seen = set()
        unique_discovered = []
        for plugin in discovered:
            if plugin not in seen:
                seen.add(plugin)
                unique_discovered.append(plugin)

        logger.info(f"Discovered {len(unique_discovered)} plugins")
        return unique_discovered

    def load_plugin(self, plugin_name: str) -> bool:
        """
        Load a specific plugin by name.

        Args:
            plugin_name: Name of the plugin to load

        Returns:
            True if plugin loaded successfully
        """
        try:
            # Check if already loaded
            if plugin_name in self.loaded_plugins:
                logger.warning(f"Plugin {plugin_name} already loaded")
                return True

            # Import the plugin module
            module = importlib.import_module(plugin_name)

            # Extract plugin information
            plugin_info = self._extract_plugin_info(module, plugin_name)

            # Validate plugin
            if not self._validate_plugin(plugin_info):
                raise ValueError(f"Plugin validation failed for {plugin_name}")

            # Check dependencies
            if not self._check_dependencies(plugin_info):
                raise ImportError(f"Missing dependencies for plugin {plugin_name}")

            # Store plugin info
            self.plugins[plugin_name] = plugin_info
            self.loaded_plugins[plugin_name] = module

            plugin_info.loaded = True

            # Notify callback
            if self.on_plugin_loaded:
                self.on_plugin_loaded(plugin_name, plugin_info)

            logger.info(f"Plugin loaded: {plugin_name} v{plugin_info.version}")
            return True

        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")

            # Notify error callback
            if self.on_plugin_error:
                self.on_plugin_error(plugin_name, e)

            return False

    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin.

        Args:
            plugin_name: Name of the plugin to unload

        Returns:
            True if plugin unloaded successfully
        """
        try:
            if plugin_name not in self.loaded_plugins:
                logger.warning(f"Plugin {plugin_name} not loaded")
                return True

            # Get plugin info
            plugin_info = self.plugins.get(plugin_name)
            if plugin_info:
                plugin_info.loaded = False

            # Remove from loaded plugins
            del self.loaded_plugins[plugin_name]

            # Remove from plugins dict
            self.plugins.pop(plugin_name, None)

            # Notify callback
            if self.on_plugin_unloaded:
                self.on_plugin_unloaded(plugin_name)

            logger.info(f"Plugin unloaded: {plugin_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to unload plugin {plugin_name}: {e}")
            return False

    def reload_plugin(self, plugin_name: str) -> bool:
        """
        Reload a plugin.

        Args:
            plugin_name: Name of the plugin to reload

        Returns:
            True if plugin reloaded successfully
        """
        logger.info(f"Reloading plugin: {plugin_name}")

        # Unload first
        self.unload_plugin(plugin_name)

        # Clear from sys.modules to force reimport
        module_parts = plugin_name.split('.')
        for i in range(len(module_parts)):
            module_name = '.'.join(module_parts[:i+1])
            if module_name in sys.modules:
                del sys.modules[module_name]

        # Load again
        return self.load_plugin(plugin_name)

    def load_all_plugins(self) -> Dict[str, bool]:
        """
        Load all discovered plugins.

        Returns:
            Dict mapping plugin names to load success status
        """
        discovered = self.discover_plugins()
        results = {}

        for plugin_name in discovered:
            results[plugin_name] = self.load_plugin(plugin_name)

        success_count = sum(results.values())
        logger.info(f"Loaded {success_count}/{len(results)} plugins")

        return results

    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """Get information about a loaded plugin."""
        return self.plugins.get(plugin_name)

    def get_loaded_plugins(self) -> List[str]:
        """Get list of loaded plugin names."""
        return list(self.loaded_plugins.keys())

    def get_available_plugins(self) -> List[str]:
        """Get list of available plugin names."""
        return list(self.plugins.keys())

    def get_scraper_class(self, plugin_name: str) -> Optional[Type[BaseScraper]]:
        """Get the scraper class from a loaded plugin."""
        plugin_info = self.plugins.get(plugin_name)
        return plugin_info.scraper_class if plugin_info else None

    def get_config_class(self, plugin_name: str) -> Optional[Type[ScraperConfig]]:
        """Get the config class from a loaded plugin."""
        plugin_info = self.plugins.get(plugin_name)
        return plugin_info.config_class if plugin_info else None

    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin."""
        plugin_info = self.plugins.get(plugin_name)
        if plugin_info:
            plugin_info.enabled = True
            logger.info(f"Plugin enabled: {plugin_name}")
            return True
        return False

    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin."""
        plugin_info = self.plugins.get(plugin_name)
        if plugin_info:
            plugin_info.enabled = False
            logger.info(f"Plugin disabled: {plugin_name}")
            return True
        return False

    def is_plugin_enabled(self, plugin_name: str) -> bool:
        """Check if a plugin is enabled."""
        plugin_info = self.plugins.get(plugin_name)
        return plugin_info.enabled if plugin_info else False

    def _extract_plugin_info(self, module: Any, plugin_name: str) -> PluginInfo:
        """Extract plugin information from a loaded module."""
        plugin_info = PluginInfo(name=plugin_name)

        # Try to get plugin metadata
        if hasattr(module, '__version__'):
            plugin_info.version = module.__version__

        if hasattr(module, '__description__'):
            plugin_info.description = module.__description__

        if hasattr(module, '__author__'):
            plugin_info.author = module.__author__

        if hasattr(module, '__dependencies__'):
            plugin_info.dependencies = module.__dependencies__

        # Find scraper class
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and
                issubclass(obj, BaseScraper) and
                obj != BaseScraper):
                plugin_info.scraper_class = obj
                break

        # Find config class
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and
                issubclass(obj, ScraperConfig) and
                obj != ScraperConfig):
                plugin_info.config_class = obj
                break

        # Extract metadata
        plugin_info.metadata = {
            'module': module.__name__,
            'file': getattr(module, '__file__', None),
            'classes': [name for name, obj in inspect.getmembers(module, inspect.isclass)],
            'functions': [name for name, obj in inspect.getmembers(module, inspect.isfunction)]
        }

        return plugin_info

    def _validate_plugin(self, plugin_info: PluginInfo) -> bool:
        """Validate a plugin's structure."""
        if not plugin_info.scraper_class:
            logger.error(f"Plugin {plugin_info.name} does not contain a scraper class")
            return False

        # Check if scraper class is properly defined
        if not issubclass(plugin_info.scraper_class, BaseScraper):
            logger.error(f"Plugin {plugin_info.name} scraper class does not inherit from BaseScraper")
            return False

        # Check for required methods
        required_methods = ['_execute_scrape']
        for method in required_methods:
            if not hasattr(plugin_info.scraper_class, method):
                logger.error(f"Plugin {plugin_info.name} missing required method: {method}")
                return False

        return True

    def _check_dependencies(self, plugin_info: PluginInfo) -> bool:
        """Check if plugin dependencies are satisfied."""
        for dependency in plugin_info.dependencies:
            try:
                importlib.import_module(dependency)
            except ImportError:
                logger.error(f"Missing dependency for plugin {plugin_info.name}: {dependency}")
                return False

        return True

    def get_plugin_metrics(self) -> Dict[str, Any]:
        """Get plugin system metrics."""
        return {
            'total_plugins': len(self.plugins),
            'loaded_plugins': len(self.loaded_plugins),
            'enabled_plugins': sum(1 for p in self.plugins.values() if p.enabled),
            'plugin_types': {
                'scrapers': sum(1 for p in self.plugins.values() if p.scraper_class),
                'configs': sum(1 for p in self.plugins.values() if p.config_class)
            },
            'plugin_list': [
                {
                    'name': name,
                    'version': info.version,
                    'loaded': info.loaded,
                    'enabled': info.enabled,
                    'has_scraper': info.scraper_class is not None
                }
                for name, info in self.plugins.items()
            ]
        }

    def scan_for_updates(self) -> Dict[str, bool]:
        """
        Scan for plugin updates and reload changed plugins.

        Returns:
            Dict mapping plugin names to reload success status
        """
        results = {}

        for plugin_name in self.get_loaded_plugins():
            try:
                # Check if plugin file has changed (simplified check)
                plugin_info = self.plugins.get(plugin_name)
                if plugin_info and plugin_info.metadata.get('file'):
                    file_path = plugin_info.metadata['file']
                    if os.path.exists(file_path):
                        # In production, you'd check file modification time
                        # For now, just attempt reload
                        results[plugin_name] = self.reload_plugin(plugin_name)
                    else:
                        logger.warning(f"Plugin file missing: {file_path}")
                        results[plugin_name] = False
                else:
                    results[plugin_name] = True  # Assume up to date

            except Exception as e:
                logger.error(f"Error checking updates for {plugin_name}: {e}")
                results[plugin_name] = False

        updated_count = sum(results.values())
        logger.info(f"Plugin update scan complete: {updated_count}/{len(results)} updated")

        return results

    async def cleanup(self) -> None:
        """Cleanup plugin manager resources."""
        logger.info("Cleaning up PluginManager...")

        # Unload all plugins
        for plugin_name in list(self.loaded_plugins.keys()):
            self.unload_plugin(plugin_name)

        self.plugins.clear()
        self.loaded_plugins.clear()

        logger.info("PluginManager cleanup complete")
