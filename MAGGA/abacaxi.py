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
    
    net.pingAll()

    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel( 'info' )
    OVSKernelSwitch.OVSVersion = '2.15.0' 
    myNetwork()

