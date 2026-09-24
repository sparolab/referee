# Running ReFeree in Docker

Everything in this repository runs inside one image — the C++ core, its
validation tools, the Python reference they are checked against, and the ROS 2
workspace. Nothing needs to be installed on the host but Docker.

## Build the image

```bash
docker compose -f docker/docker-compose.yml build
```

`ros:humble-ros-base-jammy` plus OpenCV, PCL, Eigen, FFTW, OpenMP and GTSAM 4.2
built from source. The first build takes a while because of GTSAM; later ones
hit the layer cache.

## Run the checks

```bash
docker compose -f docker/docker-compose.yml run --rm validate
```

Rebuilds the mounted C++ core and runs, for MulRan `KAIST_03` and
`Riverside_03`: the descriptor against the archived `.npy`, the two-stage search
against the Python reference at four parameter settings, and phase-correlation
registration against synthetic ground truth. Exits non-zero if anything fails,
so it drops straight into CI.

## Interactive shell

```bash
docker compose -f docker/docker-compose.yml run --rm referee
```

Inside, the entrypoint defines:

| command | what it does |
| --- | --- |
| `referee-build-core` | rebuild the header-only C++ core and its tools |
| `referee-build` | `colcon build` the ROS 2 workspace |
| `referee-validate [seq ...]` | run the checks, default `KAIST_03 Riverside_03` |

## Mounts

Sources are bind-mounted, so host edits take effect without rebuilding the
image. Datasets are mounted read-only.

| host | container | |
| --- | --- | --- |
| `cpp/`, `docker/`, `ros2_ws/` | `/referee/...` | read-write |
| `/storage/Datasets/ReFeree` | `/data/referee` | read-only |
| `/nas2/Radar_Datasets` | `/data/radar` | read-only |

Point them elsewhere with environment variables:

```bash
REFEREE_MULRAN_HOST=/mnt/referee REFEREE_RADAR_HOST=/mnt/radar \
  docker compose -f docker/docker-compose.yml run --rm validate
```

`/nas2` is an NFS mount of a NAS on the lab network, so it is only reachable
from a machine on that network; `/data/radar` is simply absent otherwise, and
the checks that need it are skipped.

## Building the ROS 2 workspace

```bash
docker compose -f docker/docker-compose.yml run --rm build
```

`ros2_ws/src` is empty until the odometry and LT-SLAM packages land; the command
is a no-op until then.

## GUI

`network_mode: host` and the X11 socket are already wired up, so `rviz2` works
after allowing local connections on the host:

```bash
xhost +local:docker
```
