def get_model(args):
    model_name = args.model_name.lower()

    if model_name == "ncm":
        from .ncm import NCM
        return NCM(device=args.device)

    elif model_name == "md":
        from .md import MD
        return MD(reg=args.reg, device=args.device)

    elif model_name == "klda":
        from .klda import KLDA
        return KLDA(D=args.D, gamma=args.gamma, reg=args.reg, seed=args.seed, device=args.device)

    elif model_name == "residuals":
        from .residuals import Residuals
        return Residuals(n_comp=args.n_comp, device=args.device)

    elif model_name == "neco":
        from .neco import NECO
        return NECO(fc_path=args.fc_path, n_comp=args.n_comp, device=args.device)

    elif model_name == "ananc":
        from .ananc import AnaNC
        return AnaNC(D=args.D, reg=args.reg, seed=args.seed, device=args.device)
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")
