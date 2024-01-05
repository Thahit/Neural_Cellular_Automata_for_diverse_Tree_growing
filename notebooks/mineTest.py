import grpc
from mcipc.query import Client
from test_evocraft_py import minecraft_pb2_grpc, FillCubeRequest, Cube, Point, AIR

with Client('127.0.0.1', 25565) as client:
    basic_stats = client.stats()            # Get basic stats.
    full_stats = client.stats(full=True)
print(basic_stats)
print(full_stats)
# print(mansion)
# print(badlands)

channel = grpc.insecure_channel('localhost:25565')
client = minecraft_pb2_grpc.MinecraftServiceStub(channel)
client.fillCube(FillCubeRequest(  # Clear a 20x10x20 working area
    cube=Cube(
        min=Point(x=-10, y=4, z=-10),
        max=Point(x=10, y=14, z=10)
    ),
    type=AIR
))