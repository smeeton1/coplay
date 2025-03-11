#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>

// g++ movable_button.cpp -o movable_button -lsfml-graphics -lsfml-window -lsfml-system

int main() {
    // Create a window
    sf::RenderWindow window(sf::VideoMode(400, 300), "My First Window");
    // Create a button (a rectangle shape)
    sf::RectangleShape button(sf::Vector2f(100, 50));
    button.setFillColor(sf::Color::Green);
    button.setPosition(150, 125); // Initial position
    // Main loop
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
            // Check for mouse button press
            if (event.type == sf::Event::MouseButtonPressed) {
                if (event.mouseButton.button == sf::Mouse::Left) {
                    // Check if the mouse is over the button
                    if (button.getGlobalBounds().contains(event.mouseButton.x, event.mouseButton.y)) {
                        window.close(); // Close the window if the button is clicked
                    }
                }
            }
        }
        // Handle keyboard input
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up)) {
            if (button.getPosition().y > 0) { // Check upper boundary
                button.move(0, -1); // Move up
            }
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down)) {
            if (button.getPosition().y + button.getSize().y < window.getSize().y) { // Check lower boundary
                button.move(0, 1); // Move down
            }
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left)) {
            if (button.getPosition().x > 0) { // Check left boundary
                button.move(-1, 0); // Move left
            }
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) {
            if (button.getPosition().x + button.getSize().x < window.getSize().x) { // Check right boundary
                button.move(1, 0); // Move right
            }
        }
        // Clear the window
        window.clear(sf::Color::Black);
        
        // Draw the button
        window.draw(button);
        
        // Display the contents of the window
        window.display();
    }
    return 0;
}