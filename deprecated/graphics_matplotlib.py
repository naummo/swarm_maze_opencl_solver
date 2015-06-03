import configs as cfg


def matplotlib_animate(plot_update):
    """ Generate simulation """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    # noinspection PyUnresolvedReferences
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()

    anim = animation.FuncAnimation(fig, plot_update, fargs=(fig,), frames=cfg.total_timesteps,
                                   interval=1 / cfg.framespersecond, blit=True)
    return anim, fig


def matplotlib_savevideo(anim):
    """ Save generated animation """
    # writer = animation.ImageMagickFileWriter("jpeg")
    # writer.setup(fig, "hi.jpg", dpi=dpi, frame_prefix=u"_tmp", clear_temp=False)
    anim.save(cfg.movie_url, dpi=cfg.dpi, fps=cfg.framespersecond, codec="mpeg4")
