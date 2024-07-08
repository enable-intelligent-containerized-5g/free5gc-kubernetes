# Pop model
class Pod:
    def __init__(self, kind, name, list_interfaces=None):
        self.kind = kind
        self.name = name
        if list_interfaces is None:
            list_interfaces = []
        self.list_interfaces = list_interfaces

    def __str__(self):
        # Convert the list of interfaces in a comma separated string
        list_interfaces_str = ', '.join(str(interface) for interface in self.list_interfaces)
        # Print the Pod
        return f"{self.kind} {self.name} with interfaces [{list_interfaces_str}]"