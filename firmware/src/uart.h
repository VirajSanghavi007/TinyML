#ifndef UART_H_
#define UART_H_

#ifdef __cplusplus
extern "C" {
#endif

// Polled transmit on RA4M1 SCI2 (Uno R4 D0/D1 header UART; Renode's console for this board).
void UartInit(void);
void UartWrite(const char* s);

#ifdef __cplusplus
}
#endif

#endif  // UART_H_
