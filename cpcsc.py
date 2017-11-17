#!/usr/bin/env python2.7
import xml.etree.ElementTree as ET
import re

# ----------------------------------------------------------------------------#


def get_node_count(file):
    # Open the .csc file in ET
    tree = ET.parse(file)
    root = tree.getroot()

    # Get all the motes
    el_motes = root.findall('.simulation/mote')
    # Return the number of motes in the simulation
    return len(el_motes)

# ----------------------------------------------------------------------------#


def append_make(file, append):
    # Open the .csc file in ET
    tree = ET.parse(file)
    root = tree.getroot()

    # Get the makefile commands
    el_commands = root.findall('.simulation/motetype/commands')
    # Print tags, attribs, and text
    for el in el_commands:
        print "> cscparser.py orig: " + el.tag, el.attrib, el.text
        cmd = re.match(".*?TARGET=\w+(\s??)", el.text).group()
        el.text = cmd + ' ' + append
        print "> cscparser.py new: " + el.tag, el.attrib, el.text
        el.set('updated', 'yes')

    tree.write(file)


# ----------------------------------------------------------------------------#
def test():
    # appendmake('/home/mike/Repos/usdn/examples/ipv6/sdn-udp/'
    #            'sdn-udp-3-node_exp5438.csc', 'RUN_CONF_WITH_SDN=1')
    get_node_count('/home/mike/Repos/usdn/examples/ipv6/sdn-udp/'
                   'sdn-udp-3-node_exp5438.csc')

# ----------------------------------------------------------------------------#


if __name__ == "__main__":
    test()
