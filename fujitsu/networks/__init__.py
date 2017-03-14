from fujitsu.networks.convnets import network1, network0, network2, network3, network4, wider_network, deeper_network


_networks = {
    'network0': network0,
    'network1': network1,
    'network2': network2,
    'network3': network3,
    'network4': network4,
    'wider_network': wider_network,
    'deeper_network': deeper_network,
}


def get_network(network_name):
    assert network_name in _networks.keys(), "Network name not recognized"
    return _networks[network_name]
