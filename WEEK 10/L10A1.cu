#include <stdio.h>
#include <cuda_runtime.h>

#define MAX_ITEMS 5
#define MAX_FRIENDS 10

// Structure to store item information
struct Item {
    char name[20];
    float price;
};

// Kernel to calculate the total price for each friend
__global__ void calculateTotal(float *d_prices, int *d_purchases, float *d_totals, int numItems, int numFriends) {
    int friendId = blockIdx.x * blockDim.x + threadIdx.x;

    if (friendId < numFriends) {
        float total = 0;
        
        // Loop over the items selected by this friend
        for (int i = 0; i < numItems; i++) {
            total += d_prices[i] * d_purchases[friendId * numItems + i];  // Multiply price by the quantity purchased
        }
        
        d_totals[friendId] = total;  // Store total for this friend
    }
}

void displayMenu(Item *items, int numItems) {
    printf("Shopping Mall Menu:\n");
    for (int i = 0; i < numItems; i++) {
        printf("%d. %s - $%.2f\n", i + 1, items[i].name, items[i].price);
    }
}

int main() {
    // Define item menu
    Item items[MAX_ITEMS] = {
        {"Shirt", 20.5f},
        {"Jeans", 30.0f},
        {"Shoes", 50.0f},
        {"Hat", 15.0f},
        {"Sunglasses", 25.0f}
    };
    
    int numFriends;
    printf("Enter the number of friends (max %d): ", MAX_FRIENDS);
    scanf("%d", &numFriends);

    // Ensure valid number of friends
    if (numFriends <= 0 || numFriends > MAX_FRIENDS) {
        printf("Invalid number of friends!\n");
        return -1;
    }

    // Display menu
    displayMenu(items, MAX_ITEMS);
    
    // Allocate memory for purchases and totals
    int *purchases = (int *)malloc(numFriends * MAX_ITEMS * sizeof(int));  // Purchases for each friend
    float *totals = (float *)malloc(numFriends * sizeof(float));  // Total purchase per friend
    float *prices = (float *)malloc(MAX_ITEMS * sizeof(float));  // Item prices
    
    for (int i = 0; i < MAX_ITEMS; i++) {
        prices[i] = items[i].price;  // Store item prices in array
    }

    // Gather purchases from each friend
    printf("\nEnter the quantity of each item purchased by each friend:\n");
    for (int i = 0; i < numFriends; i++) {
        printf("\nFriend %d's Purchases:\n", i + 1);
        for (int j = 0; j < MAX_ITEMS; j++) {
            printf("How many %s did they buy? ", items[j].name);
            scanf("%d", &purchases[i * MAX_ITEMS + j]);
        }
    }

    // Allocate memory on device
    float *d_prices, *d_totals;
    int *d_purchases;

    cudaMalloc((void**)&d_prices, MAX_ITEMS * sizeof(float));
    cudaMalloc((void**)&d_purchases, numFriends * MAX_ITEMS * sizeof(int));
    cudaMalloc((void**)&d_totals, numFriends * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_prices, prices, MAX_ITEMS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_purchases, purchases, numFriends * MAX_ITEMS * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel to calculate totals
    int blockSize = 128;  // Number of threads per block
    int gridSize = (numFriends + blockSize - 1) / blockSize;  // Number of blocks

    calculateTotal<<<gridSize, blockSize>>>(d_prices, d_purchases, d_totals, MAX_ITEMS, numFriends);

    // Copy result back from device to host
    cudaMemcpy(totals, d_totals, numFriends * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the totals for each friend
    printf("\nTotal Purchases by Each Friend:\n");
    for (int i = 0; i < numFriends; i++) {
        printf("Friend %d: $%.2f\n", i + 1, totals[i]);
    }

    // Clean up memory
    free(purchases);
    free(totals);
    free(prices);
    cudaFree(d_prices);
    cudaFree(d_purchases);
    cudaFree(d_totals);

    return 0;
}
