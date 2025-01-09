#!/usr/bin/env python

from mininet.net import Mininet
from mininet.node import Controller, RemoteController, OVSController, OVSSwitch
from mininet.node import CPULimitedHost, Host, Node
from mininet.node import OVSKernelSwitch, UserSwitch
from mininet.node import IVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink, Intf
from subprocess import call
import time
import subprocess

def myNetwork():

    info( '*** Adding controller\n' )
    #c0 = net.addController(name='c0', controller=RemoteController, switch=OVSKernelSwitch, ip='127.0.0.1', port=6653)
    c0 = RemoteController('ryu', ip='127.0.0.1', port=6633)
    net = Mininet(topo=None, controller=c0, switch=OVSSwitch, link=TCLink)

    info( '*** Add switches\n')
    s1 = net.addSwitch('s1', cls=OVSKernelSwitch)
    s2 = net.addSwitch('s2', cls=OVSKernelSwitch)
    s3 = net.addSwitch('s3', cls=OVSKernelSwitch)
    s4 = net.addSwitch('s4', cls=OVSKernelSwitch)
    s5 = net.addSwitch('s5', cls=OVSKernelSwitch)
    s6 = net.addSwitch('s6', cls=OVSKernelSwitch)
    s7 = net.addSwitch('s7', cls=OVSKernelSwitch)
    s8 = net.addSwitch('s8', cls=OVSKernelSwitch)
    s9 = net.addSwitch('s9', cls=OVSKernelSwitch)
    s10 = net.addSwitch('s10', cls=OVSKernelSwitch)
    s11 = net.addSwitch('s11', cls=OVSKernelSwitch)

    info( '*** Add hosts\n')
    h1 = net.addHost('h1', cls=Host, ip='10.0.0.1', defaultRoute=None)
    h2 = net.addHost('h2', cls=Host, ip='10.0.0.2', defaultRoute=None)
    h3 = net.addHost('h3', cls=Host, ip='10.0.0.3', defaultRoute=None)
    h4 = net.addHost('h4', cls=Host, ip='10.0.0.4', defaultRoute=None)

    info( '*** Add links\n')
    net.addLink(h1, s2)
    net.addLink(s3, s4)
    net.addLink(s4, h4)
    net.addLink(h4, s3)
    net.addLink(s8, s7)
    net.addLink(s7, h3)
    net.addLink(h3, s8)
    net.addLink(s5, s6)
    net.addLink(s6, h2)
    net.addLink(s5, h2)
    net.addLink(s7, s6)
    net.addLink(h1, s1)
    net.addLink(s1, s2)
    net.addLink(s1, s9)
    net.addLink(s2, s9)
    net.addLink(s3, s10)
    net.addLink(s10, s4)
    net.addLink(s10, s9)
    net.addLink(s9, s11)
    net.addLink(s11, s10)
    net.addLink(s7, s11)
    net.addLink(s11, s6)
    net.addLink(s5, s11)
    net.addLink(s11, s8)
    net.addLink(s10, s8)
    net.addLink(s9, s5)
    net.addLink(s1, s3)
    net.addLink(s4, s8)
    net.addLink(s2, s5)

    info( '*** Starting network\n')
    net.build()
    info( '*** Starting controllers\n')
    for controller in net.controllers:
        controller.start()

    info( '*** Starting switches\n')
    net.get('s1').start([c0])
    net.get('s2').start([c0])
    net.get('s3').start([c0])
    net.get('s4').start([c0])
    net.get('s5').start([c0])
    net.get('s6').start([c0])
    net.get('s7').start([c0])
    net.get('s8').start([c0])
    net.get('s9').start([c0])
    net.get('s10').start([c0])
    net.get('s11').start([c0])

    info( '*** Post configure switches and hosts\n')

    # Test connectivity
    net.pingAll()

    return net, h1, h2, h3

def start_http_server(host):
    "Start an HTTP server on the host."
    print(f"Starting HTTP server on {host}")
    host.cmd('python3 -m http.server 80 &')

def make_http_request(client, server_ip):
    "Make an HTTP request from the client to the server."
    print(f"{client} making HTTP request to {server_ip}")
    start_time = time.time()
    result = client.cmd(f'curl http://{server_ip}:80')
    end_time = time.time()
    response_time = end_time - start_time
    print(f"Response time: {response_time} seconds")
    return response_time

def generate_congestion(host):
    "Generate traffic with iperf3 to simulate congestion."
    print(f"Generating traffic from {host}")
    host.cmd('iperf3 -s &')
    time.sleep(1)  # Let server start
    host.cmd('iperf3 -c 10.0.0.1 -t 60 -P 10 &')  # Start congestion for 60 seconds

def scenario_1(net, h1, h2, h3):
    "Scenario 1: Network without recalculation algorithm"
    print("Running Scenario 1 (No Path Recalculation)...")

    # Start HTTP server on h1
    start_http_server(h1)

    # Generate congestion using h3
    generate_congestion(h3)

    # Send HTTP request from h2 to h1
    response_time = make_http_request(h2, h1.IP())

    return response_time

def scenario_2(net, h1, h2, h3):
    "Scenario 2: Network with path recalculation algorithm"
    print("Running Scenario 2 (With Path Recalculation)...")

    # Start HTTP server on h1
    start_http_server(h1)

    # Generate congestion using h3
    generate_congestion(h3)

    # Send HTTP request from h2 to h1
    response_time = make_http_request(h2, h1.IP())

    return response_time

if __name__ == '__main__':
    # Create topology
    setLogLevel( 'info' )
    OVSKernelSwitch.OVSVersion = '2.15.0' 
    
    net, h1, h2, h3 = myNetwork()

    # Scenario 1: Without path recalculation
    response_time_1 = scenario_1(net, h1, h2, h3)
    print(f"Scenario 1 Response Time: {response_time_1} seconds")

    # Simulate path recalculation algorithm (You can integrate your OpenFlow controller here)
    #print("Activating Path Recalculation Algorithm...")
    #time.sleep(10)  # Simulate some time for the algorithm to activate

    # Scenario 2: With path recalculation
    #response_time_2 = scenario_2(net, h1, h2, h3)
    #print(f"Scenario 2 Response Time: {response_time_2} seconds")

    # Compare results
    #print(f"Scenario 1 (No Recalc) vs Scenario 2 (With Recalc):")
    #print(f"Response Time Improvement: {response_time_1 - response_time_2} seconds")

    # Drop into Mininet CLI for further testing
    CLI(net)

    # Stop network
    net.stop()

