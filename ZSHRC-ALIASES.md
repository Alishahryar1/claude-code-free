# Add to ~/.zshrc

# Single alias for launching Claude Code with proxy
alias c='just claude'

# Optional: if you use this project from other directories
# Add function to wrap just with cd
fcc() {
    cd ~/Projects/liberated-claude-code && just "$@"
}
