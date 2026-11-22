# Record the hardware / software environment of the measurements (writes results/environment.txt).
# Run it with Docker Desktop started and the cluster stopped.
param([string]$Out = (Join-Path $PSScriptRoot "..\results\environment.txt"))
$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force (Split-Path $Out) | Out-Null
$cpu = Get-CimInstance Win32_Processor
$cs  = Get-CimInstance Win32_ComputerSystem
$os  = Get-CimInstance Win32_OperatingSystem
$mem = Get-CimInstance Win32_PhysicalMemory
$bat = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
$image = "apache/spark:3.5.6-java17"
$lines = @()
$lines += "date_utc: " + (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$lines += "machine: " + $cs.Manufacturer.Trim() + " " + $cs.Model.Trim() + " (" + $(if ($bat) { "laptop, on AC power during the measurements" } else { "desktop" }) + ")"
$lines += "cpu_model: " + $cpu.Name.Trim()
$lines += "physical_cores: " + $cpu.NumberOfCores
$lines += "logical_processors: " + $cpu.NumberOfLogicalProcessors
$lines += "cpu_max_clock_mhz (reported by Windows): " + $cpu.MaxClockSpeed
$lines += "l2_cache_kb (Win32_Processor): " + $cpu.L2CacheSize
$lines += "l3_cache_kb (Win32_Processor): " + $cpu.L3CacheSize
$lines += "ram_total_gb: " + [math]::Round($cs.TotalPhysicalMemory / 1GB, 2)
$lines += "ram_modules: " + (($mem | ForEach-Object { "$([math]::Round($_.Capacity/1GB))GB@$($_.ConfiguredClockSpeed)MT/s" }) -join ", ")
$lines += "os: " + $os.Caption + " " + $os.Version + " build " + $os.BuildNumber
$lines += "power_scheme: " + [regex]::Match((powercfg /getactivescheme | Out-String), '\(([^)]*)\)').Groups[1].Value
$lines += "docker_server: " + (docker version --format '{{.Server.Version}}')
$lines += "docker_compose: " + (docker compose version --short)
$info = docker info --format 'docker_vm_cpus: {{.NCPU}}; docker_vm_memory_bytes: {{.MemTotal}}; docker_vm_os: {{.OperatingSystem}}; kernel: {{.KernelVersion}}'
$lines += $info
$lines += "spark_image: $image (" + (docker image inspect $image --format '{{index .RepoDigests 0}}') + ")"
$lines += "jdk_in_image: " + ((docker run --rm $image java -version 2>&1 | Select-Object -First 2) -join " | ")
$lines += "spark_version: 3.5.6 (Scala 2.12); serializer: Java (default); broadcast and shuffle compression: LZ4 (default)"
$lines += "build_jdk: " + ((& java -version 2>&1 | Select-Object -First 1) -replace '"', '')
$lines += "maven: " + ((& mvn -v 2>&1 | Select-Object -First 1))
$lines += "python (uv): " + ((& uv run --with numpy --with pandas --with matplotlib python -c "import sys,numpy,pandas,matplotlib;print(sys.version.split()[0],'numpy',numpy.__version__,'pandas',pandas.__version__,'matplotlib',matplotlib.__version__)"))
$lines += "cpu_as_seen_in_container: " + ((docker run --rm $image sh -c "grep -m1 'model name' /proc/cpuinfo; nproc; free -m | sed -n 2p" 2>&1) -join " | ")
$lines += "worker_container: 1 executor JVM per container; executor heap 640m, 1 or 2 cores (--cores), master daemon 128m, worker daemon 64m, driver container heap 1g"
$lines += "network: one Docker bridge network (docker compose), containers on the same machine"
$lines | Set-Content -Encoding utf8 $Out
$lines
