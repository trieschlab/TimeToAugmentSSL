from gym.envs.registration import register

# kwargs={'env_name': "PointPush", 'goal_args': [[-5, -5], [5, 5]], 'maze_size_scaling': 4, 'random_start': False,
#        "fix_goal": True, "top_down_view": True, 'test': 'Test'}
register(
    id='tdw_empty-v1',
    entry_point='envs.simple_env:SimpleEnv',
)

register(
    id='tdw_room-v1',
    entry_point='envs.room_objects:RoomObjects',
)

register(
    id='tdw_toy-v1',
    entry_point='envs.toy:Toy',
)

register(
    id='tdw_six-v1',
    entry_point='envs.six_objects:SixObjects',
)

register(
    id='tdw_toys100-v1',
    entry_point='envs.six_objects:Objects100',
)

# register(
#     id='tdw_toys20-v1',
#     entry_point='envs.six_objects:Objects20',
# )

register(
    id='tdw_toys20-v1',
    entry_point='envs.six_objects:Objects20',
    kwargs={ "img_size": 128, "height_avatar": 20}
)

register(
    id='tdw_toys20_back2-v1',
    entry_point='envs.six_objects:Objects20',
    kwargs={ "img_size": 128, "height_avatar": 20, "background": 2}
)

register(
    id='tdw_toys20_back3-v1',
    entry_point='envs.six_objects:Objects20',
    kwargs={"img_size": 128, "height_avatar": 20, "background": 3}
)

register(
    id='tdw_toys20_back3_quality_anim-v1',
    entry_point='envs.six_objects:Objects20',
    kwargs={"img_size": 128, "height_avatar": 20, "background": 3, "animations" : True, "quality": True}
)

register(
    id='tdw_toys20_back3_quality-v1',
    entry_point='envs.six_objects:Objects20',
    kwargs={"img_size": 128, "height_avatar": 20, "background": 3, "quality": True}
)


register(
    id='tdw_toys20_noback-v1',
    entry_point='envs.six_objects:Objects20',
    kwargs={"img_size": 128, "background": 0, "height_avatar": -15}
)

register(
    id='tdw_toys20_nonoise-v1',
    entry_point='envs.six_objects:Objects20',
    kwargs={"img_size": 128, "height_avatar": 20, "noise": False}
)

register(
    id='tdw_toys20_norotate-v1',
    entry_point='envs.six_objects:Objects20',
    kwargs={"img_size": 128, "height_avatar": 20, "rotate": False}
)

register(
    id='tdw_toys20_noback_norotate-v1',
    entry_point='envs.six_objects:Objects20',
    kwargs={"img_size": 128, "height_avatar": -15, "rotate": False,"background":False}
)


register(
    id='tdw_toys20_nothing-v1',
    entry_point='envs.six_objects:Objects20',
    kwargs={ "img_size": 128, "height_avatar": 20, "noise": False, "rotate":False}
)


register(
    id='tdw_toys5_noback-v1',
    entry_point='envs.six_objects:Objects5',
    kwargs={"closeness": 0.6, "img_size": 128, "background": 0, "height_avatar": -15}
)

register(
    id='tdw_toys20_noback_big_img64-v1',
    entry_point='envs.six_objects:Objects20',
    kwargs={"img_size": 64, "background": 0, "height_avatar": -15}
)

    # for close in [1.,2.,6.,1.]:
for blur in [0.1, 1, 2, 3, 5, 20]:
    i=0
    for env_name in ["tdw_toys20_back3", "tdw_toys20_noback"]:
        height_avatar = 20
        if i == 0:
            background = 3
        if i == 1:
            height_avatar = -15
            background = 0
        for str_act,num_act in {"":3, "_act5":5}.items():
            register(
                id=env_name+"_app"+str(int(blur))+str_act+"-v1",
                entry_point='envs.six_objects:Objects20',
                kwargs={"height_avatar": height_avatar, "aperture": blur, "background": background, "num_actions":num_act}
            )
            register(
                id=env_name+"_app"+str(int(blur))+str_act+"_clo1.3-v1",
                entry_point='envs.six_objects:Objects20',
                kwargs={"height_avatar": height_avatar, "aperture": blur, "background": background, "closeness": 1.3,"num_actions":num_act}
            )
            register(
                id=env_name+"_app"+str(int(blur))+str_act+"_weights-v1",
                entry_point='envs.six_objects:Objects20',
                kwargs={"height_avatar": height_avatar, "aperture": blur, "background": background, "closeness": 1., "weights": True,"num_actions":num_act}
            )
        i+=1

