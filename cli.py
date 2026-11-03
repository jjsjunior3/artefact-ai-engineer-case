from collections.abc import Callable


def run_chat(respond: Callable[[str], str]) -> None:
    print("Empório da Música | Digite 'sair' para encerrar.")
    while True:
        try:
            message = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if message.casefold() == "sair":
            break
        if message:
            print(f"Empório: {respond(message)}")


def preview_response(message: str) -> str:
    return f"[Prévia da interface] Mensagem recebida: {message}"


if __name__ == "__main__":
    run_chat(preview_response)
