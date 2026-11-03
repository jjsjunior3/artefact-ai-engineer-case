from collections.abc import Callable


def run_chat(respond: Callable[[str], str]) -> None:
    print("Oi! Sou o Léo, assistente virtual de IA da Empório da Música. Como posso ajudar?\nDigite 'sair' para encerrar.")
    while True:
        try:
            message = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if message.casefold() == "sair":
            break
        if message:
            print(f"Léo: {respond(message)}")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from agent import make_responder

    parser = argparse.ArgumentParser(description="Empório da Música chat")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("manual", type=Path)
    args = parser.parse_args()
    try:
        responder = make_responder(args.data_dir, args.manual)
    except (KeyError, OSError) as error:
        parser.exit(2, f"Configuration error: {error}\n")
    run_chat(responder)
