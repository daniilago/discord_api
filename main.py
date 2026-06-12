from collect_data import collect, collect_server_info
from analyse import print_all_filters, analyse_user
from graph.builder import build_graph
from graph.metrics import calculate_centralities, print_summary
from graph.plot import plot_network, plot_degree_distribution

if __name__ == "__main__":
    print("\n=== O QUE DESEJA ANALISAR? ===")
    print("1 - Histórico geral do canal")
    print("2 - Mensagens de um usuário específico")
    print("3 - Informações do servidor")

    opcao = input("\nEscolha: ").strip()

    if opcao == "1":
        server_name, channel_name = collect()
        print_all_filters(server_name, channel_name)
    elif opcao == "2":
        username = input("Digite o nome do usuário: ").strip()
        server_name, channel_name = collect(username)
        analyse_user(username, server_name, channel_name)
    elif opcao == "3":
        collect_server_info()
    else:
        print("Opção inválida. Encerrando.")

    G, G_undir = build_graph("[LPS] Los Pombos", "│📗│chat-geraldo")

    centralities = calculate_centralities(G_undir)
    eigen = centralities.set_index("user")["eigenvector"].to_dict()
    print_summary(G_undir)

    plot_network(G_undir, eigen, output="rede.html")
    plot_degree_distribution(G_undir, output="graus.png")