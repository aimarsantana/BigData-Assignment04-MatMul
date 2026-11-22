# Build the Spark application (bd.matmul4: kernels, distributed algorithms, benchmark driver, validation) and pull the image.
#   target/matmul4.jar is mounted read-only into the driver container (/app) by docker/docker-compose.yml.
# Needs: JDK 17+ (tested: Temurin 21.0.12, release 17), Maven 3.9, Docker Desktop with Compose.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
mvn -B -q package
if ($LASTEXITCODE -ne 0) { throw "mvn package failed" }
docker pull apache/spark:3.5.6-java17
if ($LASTEXITCODE -ne 0) { throw "docker pull failed" }
Write-Host "built target/matmul4.jar (Spark 3.5.6 is 'provided': the runtime is the one in the image)"
