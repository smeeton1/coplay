#include <SFML/Graphics.hpp>
#include <iostream>

int main()
{
    const int WINDOW_WIDTH = 300;
    const int WINDOW_HEIGHT = 200;

    sf::RenderWindow window(
        sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT),
        "Menu Screen"
    );

    // Moving button settings
    const float BUTTON_WIDTH = 100.0f;
    const float BUTTON_HEIGHT = 50.0f;

    // Movement speed in pixels per second
    const float SPEED = 100.0f;

    float buttonX = 100.0f;
    float buttonY = 75.0f;

    bool movingButton = false;

    // Load font
    sf::Font font;

    if (!font.loadFromFile("arial.ttf"))
    {
        std::cout << "Could not load arial.ttf\n";
        return 1;
    }

    // Clock used to calculate frame time
    sf::Clock clock;

    while (window.isOpen())
    {
        // Time since the previous frame
        float deltaTime = clock.restart().asSeconds();

        sf::Event event;

        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
            {
                window.close();
            }

            // -------------------------
            // MOUSE CLICK
            // -------------------------

            if (event.type == sf::Event::MouseButtonPressed &&
                event.mouseButton.button == sf::Mouse::Left)
            {
                float mouseX = event.mouseButton.x;
                float mouseY = event.mouseButton.y;

                if (!movingButton)
                {
                    // Start button
                    if (mouseX >= 70 && mouseX <= 230 &&
                        mouseY >= 70 && mouseY <= 110)
                    {
                        movingButton = true;
                    }

                    // Close button
                    if (mouseX >= 100 && mouseX <= 200 &&
                        mouseY >= 130 && mouseY <= 170)
                    {
                        window.close();
                    }
                }
                else
                {
                    // Moving button
                    if (mouseX >= buttonX &&
                        mouseX <= buttonX + BUTTON_WIDTH &&
                        mouseY >= buttonY &&
                        mouseY <= buttonY + BUTTON_HEIGHT)
                    {
                        movingButton = false;

                        buttonX = 100.0f;
                        buttonY = 75.0f;
                    }
                }
            }
        }

        // -------------------------
        // MOVE BUTTON
        // -------------------------

        if (movingButton)
        {
            // 100 pixels per second
            float movement = SPEED * deltaTime;

            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up))
            {
                buttonY -= movement;
            }

            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down))
            {
                buttonY += movement;
            }

            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
            {
                buttonX -= movement;
            }

            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
            {
                buttonX += movement;
            }

            // Keep button inside window
            if (buttonX < 0)
                buttonX = 0;

            if (buttonY < 0)
                buttonY = 0;

            if (buttonX + BUTTON_WIDTH > WINDOW_WIDTH)
                buttonX = WINDOW_WIDTH - BUTTON_WIDTH;

            if (buttonY + BUTTON_HEIGHT > WINDOW_HEIGHT)
                buttonY = WINDOW_HEIGHT - BUTTON_HEIGHT;
        }

        // -------------------------
        // DRAW
        // -------------------------

        window.clear(sf::Color(240, 240, 240));

        if (!movingButton)
        {
            // =========================
            // MENU
            // =========================

            sf::Text title;
            title.setFont(font);
            title.setString("Menu Screen");
            title.setCharacterSize(24);
            title.setFillColor(sf::Color::Black);
            title.setPosition(75, 20);

            window.draw(title);

            // Start button
            sf::RectangleShape startButton(
                sf::Vector2f(160, 40)
            );

            startButton.setPosition(70, 70);
            startButton.setFillColor(
                sf::Color(70, 130, 220)
            );

            window.draw(startButton);

            sf::Text startText;
            startText.setFont(font);
            startText.setString("Start");
            startText.setCharacterSize(16);
            startText.setFillColor(sf::Color::White);
            startText.setPosition(130, 80);

            window.draw(startText);

            // Close button
            sf::RectangleShape closeButton(
                sf::Vector2f(100, 40)
            );

            closeButton.setPosition(100, 130);
            closeButton.setFillColor(
                sf::Color(200, 70, 70)
            );

            window.draw(closeButton);

            sf::Text closeText;
            closeText.setFont(font);
            closeText.setString("Close");
            closeText.setCharacterSize(16);
            closeText.setFillColor(sf::Color::White);
            closeText.setPosition(130, 140);

            window.draw(closeText);
        }
        else
        {
            // =========================
            // MOVING BUTTON
            // =========================

            sf::RectangleShape button(
                sf::Vector2f(
                    BUTTON_WIDTH,
                    BUTTON_HEIGHT
                )
            );

            button.setPosition(buttonX, buttonY);
            button.setFillColor(
                sf::Color(70, 130, 220)
            );

            window.draw(button);

            sf::Text buttonText;
            buttonText.setFont(font);
            buttonText.setString("Click Me");
            buttonText.setCharacterSize(16);
            buttonText.setFillColor(sf::Color::White);

            buttonText.setPosition(
                buttonX + 25,
                buttonY + 15
            );

            window.draw(buttonText);

            // Instructions
            sf::Text instructions;
            instructions.setFont(font);
            instructions.setString(
                "Arrow keys to move | Click button to return"
            );
            instructions.setCharacterSize(10);
            instructions.setFillColor(sf::Color::Black);
            instructions.setPosition(35, 180);

            window.draw(instructions);
        }

        window.display();
    }

    return 0;
}