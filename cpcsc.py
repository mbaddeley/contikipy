#!/usr/bin/python
"""Contiki csc parser."""
import xml.etree.ElementTree as ET
import re


# ----------------------------------------------------------------------------#
def set_simulation_title(file, title):
    """Get the number of nodes in the csc."""
    # Open the .csc file in ET
    tree = ET.parse(file)
    root = tree.getroot()

    # Get all the motes
    el_title = root.findall('.simulation/title')
    for el in el_title:
        print('> cpcsc orig: ' + el.tag, el.attrib, el.text)
        el.text = title
        print('> cpcsc new: ' + el.tag, el.attrib, el.text)
        # el.set('updated', 'yes')

    tree.write(file)


# ----------------------------------------------------------------------------#
def get_node_count(file):
    """Get the number of nodes in the csc."""
    # Open the .csc file in ET
    tree = ET.parse(file)
    root = tree.getroot()

    # Get all the motes
    el_motes = root.findall('.simulation/mote')
    # Return the number of motes in the simulation
    return len(el_motes)


# ----------------------------------------------------------------------------#
def append_make(file, append):
    """Append the make command in the csc."""
    # Open the .csc file in ET
    tree = ET.parse(file)
    root = tree.getroot()

    # Get the makefile commands
    el_commands = root.findall('.simulation/motetype/commands')
    # Print tags, attribs, and text
    for el in el_commands:
        print('> cpcsc orig: ' + el.tag, el.attrib, el.text)
        cmd = re.match(".*?TARGET=\\w+(\\s??)", el.text).group()
        el.text = cmd + ' ' + append
        print('> cpcsc new: ' + el.tag, el.attrib, el.text)
        el.set('updated', 'yes')

    tree.write(file)


# ----------------------------------------------------------------------------#
def test():
    """Test this script."""
    # appendmake('/home/mike/Repos/usdn/examples/ipv6/sdn-udp/'
    #            'sdn-udp-3-node_exp5438.csc', 'RUN_CONF_WITH_SDN=1')
    get_node_count('/home/mike/Repos/usdn/examples/ipv6/sdn-udp/'
                   'sdn-udp-3-node_exp5438.csc')


# ----------------------------------------------------------------------------#
if __name__ == "__main__":
    test()
