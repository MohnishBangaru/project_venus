#!/usr/bin/env python3
"""
UI-Venus Mobile Crawler CLI

Main command-line interface for the UI-Venus Mobile Crawler,
including local and RunPod deployment options.
"""

import asyncio
import sys
import logging
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table

from config import ProjectConfig

# Import RunPod crawler with error handling
try:
    from scripts.runpod_crawler import RunPodCrawler
except ImportError as e:
    console.print(f"[yellow]Warning: RunPod crawler not available: {e}[/yellow]")
    RunPodCrawler = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

console = Console()


@click.group()
@click.option('--config', '-c', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """UI-Venus Mobile Crawler - Intelligent Android App Crawling with AI"""
    
    # Set up logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    ctx.ensure_object(dict)
    try:
        if config:
            ctx.obj['config'] = ProjectConfig.load_from_file(config)
        else:
            ctx.obj['config'] = ProjectConfig.load_from_env()
    except Exception as e:
        console.print(f"[red]Failed to load configuration: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--app', '-a', default='com.android.settings', help='Target app package name')
@click.option('--max-actions', type=int, default=100, help='Maximum number of actions')
@click.option('--max-time', type=int, default=30, help='Maximum time in minutes')
@click.option('--strategy', default='priority_based', 
              type=click.Choice(['priority_based', 'breadth_first', 'depth_first', 'random']),
              help='Crawling strategy')
@click.option('--coverage-threshold', type=float, default=0.3, help='Coverage threshold (0.0-1.0)')
@click.option('--output-dir', help='Output directory for results')
@click.pass_context
def crawl(ctx, app, max_actions, max_time, strategy, coverage_threshold, output_dir):
    """Run a crawling session on local device"""
    
    console.print(Panel.fit(
        f"[bold blue]UI-Venus Mobile Crawler[/bold blue]\n"
        f"Target App: {app}\n"
        f"Max Actions: {max_actions}\n"
        f"Max Time: {max_time} minutes\n"
        f"Strategy: {strategy}\n"
        f"Coverage Threshold: {coverage_threshold}",
        title="Crawling Configuration"
    ))
    
    # Update configuration with CLI options
    config = ctx.obj['config']
    config.device.target_package = app
    config.crawler.max_actions = max_actions
    config.crawler.max_time_minutes = max_time
    config.crawler.strategy = strategy
    config.crawler.coverage_threshold = coverage_threshold
    
    if output_dir:
        config.crawler.output_directory = output_dir
    
    # Run crawling session
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("Starting crawling session...", total=None)
        
        try:
            # Import and run local crawler
            from scripts.local_crawler import LocalCrawler
            
            crawler = LocalCrawler(config)
            
            progress.update(task, description="Initializing crawler...")
            asyncio.run(crawler.initialize())
            
            progress.update(task, description="Running crawling session...")
            results = asyncio.run(crawler.run_crawling_session())
            
            progress.update(task, description="Crawling completed!", completed=True)
            
            # Display results
            display_results(results)
            
        except Exception as e:
            console.print(f"[red]Crawling failed: {e}[/red]")
            sys.exit(1)


@cli.command()
@click.option('--app', '-a', default='com.android.settings', help='Target app package name')
@click.option('--max-actions', type=int, default=100, help='Maximum number of actions')
@click.option('--max-time', type=int, default=30, help='Maximum time in minutes')
@click.option('--adb-host', default='localhost', help='ADB host address')
@click.option('--adb-port', type=int, default=5037, help='ADB port number')
@click.option('--gpu-memory', type=float, default=0.8, help='GPU memory usage (0.0-1.0)')
@click.pass_context
def runpod(ctx, app, max_actions, max_time, adb_host, adb_port, gpu_memory):
    """Run crawling session on RunPod with remote ADB connection"""
    
    if not RunPodCrawler:
        console.print("[red]❌ RunPod crawler not available. Please check your installation.[/red]")
        sys.exit(1)
    
    console.print(Panel.fit(
        f"[bold blue]UI-Venus Mobile Crawler - RunPod Mode[/bold blue]\n"
        f"Target App: {app}\n"
        f"Max Actions: {max_actions}\n"
        f"Max Time: {max_time} minutes\n"
        f"ADB Host: {adb_host}:{adb_port}\n"
        f"GPU Memory: {gpu_memory}",
        title="RunPod Configuration"
    ))
    
    # Update configuration for RunPod
    config = ctx.obj['config']
    config.device.target_package = app
    config.device.remote_adb_host = adb_host
    config.device.remote_adb_port = adb_port
    config.device.enable_remote_adb = True
    config.device.is_runpod_environment = True
    config.crawler.max_actions = max_actions
    config.crawler.max_time_minutes = max_time
    config.ui_venus.max_memory_usage = gpu_memory
    
    # Run RunPod crawling session
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("Starting RunPod crawling session...", total=None)
        
        try:
            crawler = RunPodCrawler(config)
            
            progress.update(task, description="Setting up RunPod configurations...")
            crawler.setup_runpod_configurations(
                target_app=app,
                max_actions=max_actions,
                max_time_minutes=max_time,
                adb_host=adb_host,
                adb_port=adb_port
            )
            
            progress.update(task, description="Setting up ADB server...")
            if not crawler.setup_adb_server():
                raise Exception("Failed to setup ADB server")
            
            progress.update(task, description="Connecting to remote device...")
            if not crawler.connect_to_device():
                raise Exception("Failed to connect to remote device")
            
            progress.update(task, description="Initializing components...")
            if not asyncio.run(crawler.initialize_components()):
                raise Exception("Failed to initialize components")
            
            progress.update(task, description="Launching target app...")
            if not asyncio.run(crawler.launch_target_app()):
                raise Exception("Failed to launch target app")
            
            progress.update(task, description="Running crawling session...")
            results = asyncio.run(crawler.run_crawling_session())
            
            progress.update(task, description="RunPod crawling completed!", completed=True)
            
            # Display results
            display_results(results)
            
        except Exception as e:
            console.print(f"[red]RunPod crawling failed: {e}[/red]")
            sys.exit(1)
        finally:
            crawler.cleanup()


@cli.command()
@click.option('--device-id', help='Specific device ID to test')
@click.option('--test-actions', is_flag=True, help='Test basic device actions')
@click.pass_context
def setup_device(ctx, device_id, test_actions):
    """Setup and test Android device connection"""
    
    console.print(Panel.fit(
        "[bold blue]Device Setup and Testing[/bold blue]",
        title="Device Configuration"
    ))
    
    config = ctx.obj['config']
    if device_id:
        config.device.device_id = device_id
    
    try:
        from scripts.setup_device import DeviceSetup
        
        setup = DeviceSetup(config)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Setting up device...", total=None)
            
            progress.update(task, description="Connecting to device...")
            if not setup.connect_to_device():
                raise Exception("Failed to connect to device")
            
            progress.update(task, description="Testing device communication...")
            if not setup.test_device_communication():
                raise Exception("Device communication test failed")
            
            if test_actions:
                progress.update(task, description="Testing device actions...")
                if not setup.test_device_actions():
                    raise Exception("Device actions test failed")
            
            progress.update(task, description="Device setup completed!", completed=True)
        
        console.print("[green]✅ Device setup successful![/green]")
        
    except Exception as e:
        console.print(f"[red]Device setup failed: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--results-dir', help='Results directory to analyze')
@click.option('--output-format', default='pdf', type=click.Choice(['pdf', 'json', 'html']),
              help='Output format for analysis report')
@click.pass_context
def analyze(ctx, results_dir, output_format):
    """Analyze crawling results and generate reports"""
    
    console.print(Panel.fit(
        f"[bold blue]Results Analysis[/bold blue]\n"
        f"Results Directory: {results_dir or 'Default'}\n"
        f"Output Format: {output_format}",
        title="Analysis Configuration"
    ))
    
    try:
        from scripts.analyze_results import ResultsAnalyzer
        
        analyzer = ResultsAnalyzer(results_dir)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Analyzing results...", total=100)
            
            progress.update(task, description="Loading results data...", completed=20)
            data = analyzer.load_results()
            
            progress.update(task, description="Processing analysis...", completed=50)
            analysis = analyzer.analyze_data(data)
            
            progress.update(task, description="Generating report...", completed=80)
            report_path = analyzer.generate_report(analysis, output_format)
            
            progress.update(task, description="Analysis completed!", completed=100)
        
        console.print(f"[green]✅ Analysis completed! Report saved to: {report_path}[/green]")
        
    except Exception as e:
        console.print(f"[red]Analysis failed: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--test-type', default='all', 
              type=click.Choice(['all', 'gpu', 'adb', 'config', 'dependencies']),
              help='Type of test to run')
@click.pass_context
def test_ui_venus(ctx, test_type):
    """Test UI-Venus model integration"""
    
    console.print(Panel.fit(
        f"[bold blue]UI-Venus Model Testing[/bold blue]\n"
        f"Test Type: {test_type}",
        title="Testing Configuration"
    ))
    
    try:
        from scripts.test_ui_venus_mac import run_ui_venus_tests
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Running UI-Venus tests...", total=None)
            
            success = run_ui_venus_tests(test_type)
            
            progress.update(task, description="UI-Venus tests completed!", completed=True)
        
        if success:
            console.print("[green]✅ All UI-Venus tests passed![/green]")
        else:
            console.print("[red]❌ Some UI-Venus tests failed![/red]")
            sys.exit(1)
        
    except Exception as e:
        console.print(f"[red]UI-Venus testing failed: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--test-type', default='all',
              type=click.Choice(['all', 'environment', 'gpu', 'adb', 'workspace']),
              help='Type of RunPod test to run')
@click.pass_context
def test_runpod(ctx, test_type):
    """Test RunPod environment setup"""
    
    console.print(Panel.fit(
        f"[bold blue]RunPod Environment Testing[/bold blue]\n"
        f"Test Type: {test_type}",
        title="RunPod Testing"
    ))
    
    try:
        from scripts.test_runpod_setup import run_all_tests
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Running RunPod tests...", total=None)
            
            success = run_all_tests()
            
            progress.update(task, description="RunPod tests completed!", completed=True)
        
        if success:
            console.print("[green]✅ All RunPod tests passed![/green]")
        else:
            console.print("[red]❌ Some RunPod tests failed![/red]")
            sys.exit(1)
        
    except Exception as e:
        console.print(f"[red]RunPod testing failed: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.pass_context
def list_devices(ctx):
    """List available Android devices"""
    
    console.print(Panel.fit(
        "[bold blue]Available Android Devices[/bold blue]",
        title="Device List"
    ))
    
    try:
        import adbutils
        
        adb = adbutils.AdbClient()
        devices = adb.device_list()
        
        if devices:
            table = Table(show_header=True, header_style="bold blue")
            table.add_column("Device ID", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Model", style="yellow")
            table.add_column("Android Version", style="magenta")
            
            for device in devices:
                try:
                    model = device.shell("getprop ro.product.model").strip()
                    android_version = device.shell("getprop ro.build.version.release").strip()
                    table.add_row(device.serial, "Connected", model, android_version)
                except:
                    table.add_row(device.serial, "Connected", "Unknown", "Unknown")
            
            console.print(table)
        else:
            console.print("[yellow]⚠️ No devices found. Make sure your emulator is running.[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Failed to list devices: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('apk_path')
@click.option('--device-id', help='Specific device ID to install on')
@click.pass_context
def install_apk(ctx, apk_path, device_id):
    """Install APK on Android device"""
    
    console.print(Panel.fit(
        f"[bold blue]APK Installation[/bold blue]\n"
        f"APK Path: {apk_path}\n"
        f"Device ID: {device_id or 'Auto-detect'}",
        title="Installation Configuration"
    ))
    
    try:
        import adbutils
        
        adb = adbutils.AdbClient()
        devices = adb.device_list()
        
        if not devices:
            console.print("[red]❌ No devices found![/red]")
            sys.exit(1)
        
        device = devices[0] if not device_id else None
        if device_id:
            for d in devices:
                if d.serial == device_id:
                    device = d
                    break
        
        if not device:
            console.print(f"[red]❌ Device {device_id} not found![/red]")
            sys.exit(1)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Installing APK...", total=None)
            
            progress.update(task, description="Installing APK...")
            device.install(apk_path)
            
            progress.update(task, description="APK installation completed!", completed=True)
        
        console.print(f"[green]✅ APK installed successfully on {device.serial}![/green]")
        
    except Exception as e:
        console.print(f"[red]APK installation failed: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('package_name')
@click.option('--device-id', help='Specific device ID to uninstall from')
@click.pass_context
def uninstall_apk(ctx, package_name, device_id):
    """Uninstall APK from Android device"""
    
    console.print(Panel.fit(
        f"[bold blue]APK Uninstallation[/bold blue]\n"
        f"Package: {package_name}\n"
        f"Device ID: {device_id or 'Auto-detect'}",
        title="Uninstallation Configuration"
    ))
    
    try:
        import adbutils
        
        adb = adbutils.AdbClient()
        devices = adb.device_list()
        
        if not devices:
            console.print("[red]❌ No devices found![/red]")
            sys.exit(1)
        
        device = devices[0] if not device_id else None
        if device_id:
            for d in devices:
                if d.serial == device_id:
                    device = d
                    break
        
        if not device:
            console.print(f"[red]❌ Device {device_id} not found![/red]")
            sys.exit(1)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Uninstalling APK...", total=None)
            
            progress.update(task, description="Uninstalling APK...")
            device.uninstall(package_name)
            
            progress.update(task, description="APK uninstallation completed!", completed=True)
        
        console.print(f"[green]✅ APK uninstalled successfully from {device.serial}![/green]")
        
    except Exception as e:
        console.print(f"[red]APK uninstallation failed: {e}[/red]")
        sys.exit(1)


def display_results(results):
    """Display crawling results in a formatted table."""
    if not results:
        console.print("[yellow]⚠️ No results to display[/yellow]")
        return
    
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Actions", str(results.get('total_actions', 0)))
    table.add_row("Coverage", f"{results.get('coverage', 0):.2%}")
    table.add_row("Duration", f"{results.get('duration', 0):.2f} seconds")
    table.add_row("Screenshots", str(results.get('screenshots_taken', 0)))
    table.add_row("Errors", str(results.get('errors', 0)))
    table.add_row("Success Rate", f"{results.get('success_rate', 0):.2%}")
    
    console.print(table)


if __name__ == '__main__':
    cli()
