
# Dhana

Dhana is a dynamic traffic management tool that leverages Software Defined Networking (SDN) to mitigate network congestion in data centers. By rerouting elephant flows and managing paths, it helps optimize network performance and efficiency.

## Prerequisites

Before running Dhana, ensure that you have the following installed:

- **Mininet**: Required for running the network simulations.
- **RYU**: A component-based SDN controller that is essential for running Dhana.

## Installing Mininet

To install Mininet, follow the official [Mininet installation guide](http://mininet.org/download/).

## Installing RYU

For recent Ubuntu versions, follow these steps to install RYU:

1. Add the necessary repository:
   ```bash
   sudo add-apt-repository ppa:deadsnakes/ppa
   ```

2. Install `python3.9` and `virtualenv`:
   ```bash
   sudo apt-get install virtualenv python3.9 python3.9-distutils
   ```

3. Create a virtual environment with Python 3.9:
   ```bash
   virtualenv -p`which python3.9` ryu-python3.9-venv
   ```

4. Activate the virtual environment:
   ```bash
   source ryu-python3.9-venv/bin/activate
   ```

5. Confirm you are in the virtual environment:
   ```bash
   echo $VIRTUAL_ENV
   ```

6. Install RYU:
   ```bash
   pip install ryu
   ```

7. Fix eventlet compatibility issues by adjusting the version:
   ```bash
   pip uninstall eventlet
   pip install eventlet==0.30.2
   ```

8. Verify that RYU is installed:
   ```bash
   ryu-manager --help
   ```

Once RYU is installed, you'll be ready to run Dhana.

## Running Dhana

Follow these steps to run Dhana:


1. Activate the RYU virtual environment:
   ```bash
   source ryu-python3.9-venv/bin/activate
   ```

2. Go to the Dhana folder and start the RYU controller with Dhana:
   ```bash
   ryu-manager --observe-links Dhana.py
   ```

3. In a new terminal, navigate to the `Dhana` folder and start the network topology:
   ```bash
   sudo python ./abacaxi.py
   ```

### Done!

Dhana is now up and running! You can begin simulating network traffic and monitor how Dhana optimizes your network performance.
