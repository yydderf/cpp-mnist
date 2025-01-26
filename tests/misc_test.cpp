#include <gtest/gtest.h>

#include "../src/misc.hpp"

// see https://www.desmos.com/calculator/ahyb796dkf for function graphs
// sigmoid prime is a symmetric function, use EXPECT_NEAR due to floating point precision issue

TEST(SigmoidTest, BasicAssertions)
{
    EXPECT_NEAR(sigmoid(0), 0.5, 0);
    EXPECT_NEAR(sigmoid(1), 0.73106, 0.00001);
    EXPECT_NEAR(sigmoid(-1), 0.26894, 0.00001);
    EXPECT_NEAR(sigmoid(10), 1, 0.0001);
    EXPECT_NEAR(sigmoid(-10), 0, 0.0001);
}

TEST(SigmoidPrimeTest, BasicAssertions)
{
    EXPECT_NEAR(sigmoid_prime(0), 0.25, 0);
    EXPECT_NEAR(sigmoid_prime(1), 0.19661, 0.00001);
    EXPECT_NEAR(sigmoid_prime(-1), 0.19661, 0.00001);
    EXPECT_NEAR(sigmoid_prime(10), 0, 0.0001);
    EXPECT_NEAR(sigmoid_prime(-10), 0, 0.0001);

    EXPECT_NEAR(sigmoid_prime(1), sigmoid_prime(-1), 0.000001);
    EXPECT_NEAR(sigmoid_prime(10), sigmoid_prime(-10), 0.0000001);
}
