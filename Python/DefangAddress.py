class Solution:
    def defangIPaddr(self, address: str) -> str:
        address = address.replace(".", "[.]", 3)
        return address
