import { fireEvent, render, screen } from '@testing-library/react';
import ChatBox from '../components/ChatBox';

describe('ChatBox', () => {
  it('sends entered message', () => {
    const onSend = vi.fn();

    render(
      <ChatBox
        messages={[]}
        loading={false}
        topK={5}
        mode="qa"
        temperature={0.2}
        onTopKChange={() => undefined}
        onModeChange={() => undefined}
        onTemperatureChange={() => undefined}
        onSend={onSend}
      />,
    );

    fireEvent.change(screen.getByPlaceholderText(/Ask about methods/i), {
      target: { value: 'Explain the main result' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Send/i }));

    expect(onSend).toHaveBeenCalledWith('Explain the main result');
  });
});
